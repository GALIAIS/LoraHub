"""LoRA adapter spectral analysis.

Runs a side-band SVD over each LoRA delta-weight matrix in a saved
checkpoint, producing a small dict of "is the adapter actually
learning anything useful" metrics:

  effective_rank — Σσ)² / Σσ². Measures how many singular directions
                   carry meaningful energy. Falls if updates are
                   collapsing onto one direction (mode collapse) or
                   the configured rank is wider than the data
                   supports.
  top1_energy    — σ₀² / Σσ². Fraction of energy in the dominant
                   direction. Climbing past ~0.4 with rank-stable
                   training usually means the adapter is over-
                   specialising to a narrow mode.
  fro_norm       — ‖ΔW‖_F = α·‖B·A‖_F. Whether the delta is growing
                   over time (still learning) or saturated.

The module is intentionally framework-light: it reads the
.safetensors file with the canonical ``safetensors`` Python lib, runs
``numpy.linalg.svd`` on each (B, A) pair, and emits one
``lora_spectrum`` event with aggregate stats. Per-layer raw stats are
included up to a configurable cap so the UI / AI advisor can drill
in without fetching the file directly.

Triggering is the API host's job: ``schedule_lora_spectrum`` accepts
a checkpoint path + step and runs the analysis on a daemon thread,
emitting through the same ``on_event`` callback the rest of the job
uses. Errors are caught and logged as a single warning event so a
malformed checkpoint never breaks the live training loop.

Config-level toggle: ``cfg.sampling.spectrum_analysis`` (default
``True`` — the cost is < 1 s per checkpoint on a typical adapter and
the signal is genuinely useful). Set to ``False`` to disable.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lorahub.core.events import EventType, TrainingEvent

_log = logging.getLogger(__name__)


# Cap the number of per-layer stats we serialise. Adapters can have
# hundreds of LoRA pairs; we only need a representative handful for
# the UI to render trends without bloating events.jsonl.
_PER_LAYER_CAP = 16


@dataclass(frozen=True, slots=True)
class LoraSpectrum:
    """Aggregate spectral summary of one LoRA checkpoint."""

    layers: int
    effective_rank: float
    top1_energy: float
    fro_norm: float
    per_layer: list[dict[str, Any]]

    def to_payload(self, *, checkpoint: Path, step: int | None) -> dict[str, Any]:
        return {
            "checkpoint": str(checkpoint),
            "step": step,
            "layers": self.layers,
            "effective_rank": self.effective_rank,
            "top1_energy": self.top1_energy,
            "fro_norm": self.fro_norm,
            "per_layer": self.per_layer,
        }


def schedule_lora_spectrum(
    checkpoint: Path,
    *,
    step: int | None,
    on_event: Callable[[TrainingEvent], None],
    job_id: str | None = None,
) -> threading.Thread:
    """Run ``analyse_checkpoint`` on a daemon thread, emit the result.

    Returns the thread object so callers can join it during shutdown
    if they want to. The caller doesn't need to wait — emissions go
    through ``on_event`` like every other training event.
    """

    def _work() -> None:
        t0 = time.monotonic()
        try:
            spectrum = analyse_checkpoint(checkpoint)
        except FileNotFoundError:
            # The checkpoint may have been moved / cleaned up between
            # the ``checkpoint_saved`` event firing and our analysis
            # starting. Don't surface that as an error to the user.
            _log.info("lora-spectrum: checkpoint vanished: %s", checkpoint)
            return
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "lora-spectrum failed for %s: %s", checkpoint, exc
            )
            on_event(
                TrainingEvent(
                    type=EventType.log,
                    payload={
                        "level": "warn",
                        "source": "lora-spectrum",
                        "message": (
                            f"LoRA spectral analysis failed: {exc}; "
                            "skipping this checkpoint"
                        ),
                    },
                    job_id=job_id,
                )
            )
            return
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if spectrum is None:
            return
        on_event(
            TrainingEvent(
                type=EventType.lora_spectrum,
                payload=spectrum.to_payload(checkpoint=checkpoint, step=step)
                | {"elapsed_ms": elapsed_ms},
                job_id=job_id,
            )
        )

    t = threading.Thread(
        target=_work,
        name=f"lora-spectrum-{step or '?'}",
        daemon=True,
    )
    t.start()
    return t


def analyse_checkpoint(path: Path) -> LoraSpectrum | None:
    """Read the safetensors file at ``path`` and compute its spectrum.

    Returns ``None`` if the file contains no LoRA matrices we can
    pair up — usually because it's a non-LoRA checkpoint (full
    fine-tune, base-model snapshot, ...) or uses naming conventions
    we don't yet recognise.
    """
    import numpy as np  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".safetensors", ".sft"}:
        # Older kohya saves sometimes also produced .ckpt; we ignore those
        # because pickle loads are too risky to do on the API host.
        return None

    pairs: list[tuple[str, str, str]] = []  # (layer, key_a, key_b)
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="numpy") as f:
        all_keys = list(f.keys())
        try:
            metadata = dict(f.metadata() or {})
        except Exception:  # noqa: BLE001
            metadata = {}
        # Group keys by stripping ``.lora_down.weight`` / ``.lora_up.weight``
        # (kohya / sd-scripts convention) and ``.lora_A.weight`` /
        # ``.lora_B.weight`` (peft convention).
        for k in all_keys:
            base, suffix = _split_lora_key(k)
            if base is None or suffix is None:
                continue
            partner_suffix = _LORA_PAIR.get(suffix)
            if partner_suffix is None:
                continue
            partner = f"{base}.{partner_suffix}.weight"
            if partner not in all_keys:
                continue
            # Always keep B (up) first, A (down) second so we know the
            # multiplication order without re-checking.
            up_key, down_key = (
                (k, partner) if suffix in _UP_SUFFIXES else (partner, k)
            )
            pairs.append((base, up_key, down_key))
        # Deduplicate — a layer is registered from both halves.
        seen: set[str] = set()
        unique_pairs: list[tuple[str, str, str]] = []
        for pair in pairs:
            if pair[0] in seen:
                continue
            seen.add(pair[0])
            unique_pairs.append(pair)
        pairs = unique_pairs
        if not pairs:
            return None

        # Some LoRA savers store a per-layer alpha as either a
        # 0-dim tensor under ``<base>.alpha`` or a string in the
        # metadata; fall back to dim-of-A when neither is present.
        per_layer: list[dict[str, Any]] = []
        eff_ranks: list[float] = []
        top1s: list[float] = []
        fro_norms: list[float] = []
        cap = _PER_LAYER_CAP

        for base, up_key, down_key in pairs:
            up = np.asarray(f.get_tensor(up_key), dtype=np.float32)
            down = np.asarray(f.get_tensor(down_key), dtype=np.float32)
            # Reshape conv kernels (e.g. 4-D) to 2-D so SVD has a
            # well-defined input — flattening trailing kernel dims is
            # the same operation kohya's own merge step performs.
            up = _flatten_to_2d(up)
            down = _flatten_to_2d(down)
            if up.shape[1] != down.shape[0]:
                # Mismatch — skip this pair and keep going.
                continue
            rank = up.shape[1]
            alpha = _resolve_alpha(base, metadata, fallback_rank=rank)
            scale = alpha / rank if rank > 0 else 1.0
            delta = scale * (up @ down)
            try:
                sigma = np.linalg.svd(
                    delta, full_matrices=False, compute_uv=False
                )
            except np.linalg.LinAlgError:
                continue
            sigma = np.asarray(sigma, dtype=np.float64)
            sum_sigma = float(sigma.sum())
            sum_sigma2 = float((sigma * sigma).sum())
            if sum_sigma2 <= 0:
                continue
            eff = (sum_sigma * sum_sigma) / sum_sigma2
            top1 = float((sigma[0] ** 2) / sum_sigma2)
            fro = float(math.sqrt(sum_sigma2))
            eff_ranks.append(eff)
            top1s.append(top1)
            fro_norms.append(fro)
            if len(per_layer) < cap:
                per_layer.append(
                    {
                        "layer": base,
                        "rank": rank,
                        "alpha": alpha,
                        "effective_rank": eff,
                        "top1_energy": top1,
                        "fro_norm": fro,
                    }
                )

    if not eff_ranks:
        return None

    # Geometric mean for effective_rank because the per-layer values
    # are multiplicative (a layer at rank 16 vs rank 4 should not
    # dominate the average just by being numerically larger).
    geo_eff = math.exp(
        sum(math.log(max(r, 1e-9)) for r in eff_ranks) / len(eff_ranks)
    )
    return LoraSpectrum(
        layers=len(eff_ranks),
        effective_rank=geo_eff,
        top1_energy=sum(top1s) / len(top1s),
        fro_norm=sum(fro_norms) / len(fro_norms),
        per_layer=per_layer,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

# kohya / sd-scripts: ``<layer>.lora_down.weight`` + ``<layer>.lora_up.weight``
# peft / diffusers:    ``<layer>.lora_A.weight`` + ``<layer>.lora_B.weight``
_DOWN_SUFFIXES = ("lora_down", "lora_A")
_UP_SUFFIXES = ("lora_up", "lora_B")
_LORA_PAIR = {
    "lora_down": "lora_up",
    "lora_up": "lora_down",
    "lora_A": "lora_B",
    "lora_B": "lora_A",
}


def _split_lora_key(key: str) -> tuple[str | None, str | None]:
    """Return ``(base_layer, suffix)`` for a recognised LoRA key, else (None, None)."""
    if not key.endswith(".weight"):
        return None, None
    stem = key[: -len(".weight")]
    for suffix in (*_DOWN_SUFFIXES, *_UP_SUFFIXES):
        marker = f".{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)], suffix
    return None, None


def _resolve_alpha(
    base: str, metadata: dict[str, str], *, fallback_rank: int
) -> float:
    """Best-effort lookup of the per-layer LoRA alpha.

    Order:
      1. A separate ``<base>.alpha`` tensor — kohya-old style. We can't
         read it from this helper (no file handle), so callers may pass
         a metadata dict that already contains ``alpha`` info.
      2. Single global ``ss_network_alpha`` in the safetensors metadata
         (kohya-new). We treat this as the alpha for every layer.
      3. Fall back to the rank itself, which corresponds to scale = 1.0
         and is the worst-case assumption — the spectrum still reflects
         the relative energies between layers.
    """
    g = metadata.get("ss_network_alpha")
    if g is not None:
        try:
            return float(g)
        except ValueError:
            pass
    # Some adapters stash alpha as a JSON string (e.g. peft).
    if "ss_network_args" in metadata:
        import json  # noqa: PLC0415

        with _suppress_json():
            obj = json.loads(metadata["ss_network_args"])
            if isinstance(obj, dict):
                a = obj.get("alpha")
                if a is not None:
                    return float(a)
    return float(fallback_rank)


def _suppress_json() -> Any:
    import contextlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    return contextlib.suppress(json.JSONDecodeError, ValueError, TypeError)


def _flatten_to_2d(arr: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim < 2:
        return a.reshape(a.shape + (1,) * (2 - a.ndim))
    # 4-D conv weights — flatten trailing dims.
    return a.reshape(a.shape[0], -1)


def is_enabled(cfg: Any) -> bool:
    """Read the config toggle. Defaults to True if absent."""
    sampling = getattr(cfg, "sampling", None)
    if sampling is None:
        return True
    val = getattr(sampling, "spectrum_analysis", True)
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() not in {"false", "0", "no", "off"}
    return bool(val)


def is_lora_checkpoint(path: Path) -> bool:
    """Quick filename heuristic — only run SVD on adapter-shaped files."""
    if not path.is_file():
        return False
    name = path.name.lower()
    if name.endswith((".safetensors", ".sft")):
        # Adapters tend to be < 500 MB; full DiT snapshots are 6+ GB.
        # The threshold is loose — we'd rather skip a few full checkpoints
        # than block the queue analysing one.
        return os.path.getsize(path) < 1_500_000_000
    return False
