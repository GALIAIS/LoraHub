"""DCW v4 fusion-head data loading: bench runs + text features.

Pulls per-(stem, seed) gap trajectories from `gaps_per_sample.npz` written
by `scripts/dcw/measure_bias.py`, and per-stem text features from the
cached `{stem}_anima_te.safetensors` sidecars under `post_image_dataset/lora/`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

from library.datasets.buckets import DCW_ASPECT_TABLE


@dataclass
class Row:
    run_id: str
    aspect_id: int
    stem: str
    seed_idx: int
    gap_LL: np.ndarray  # (n_steps,) — used for target (residual on tail)
    v_rev_LL: np.ndarray  # (n_steps,) — used for input g_obs
    v_rev_source: str  # "native" | "synthetic" | "fallback"
    sigma_i: np.ndarray  # (n_steps,) — σ schedule for the run; per-row for LSQ targets
    # LL-only λ baked into the reverse trajectory at collection time
    # (one_minus_sigma schedule). 0.0 = legacy no-DCW baseline; non-zero ⇒
    # gap_LL / v_rev_LL are residuals on top of that scalar baseline, and the
    # head's α̂ is the residual.
    baseline_lambda: float
    # 2-band FEI low-band energy ∈ [0,1] captured on the latent entering each
    # reverse step (matches inference set_fei timing). None for legacy pools
    # collected before the FEI capture landed; downstream trainer code must
    # filter rows when --fei_obs != off.
    fei_low: np.ndarray | None = None


def load_bench_runs(
    results_roots: Path | list[Path],
    *,
    require_cfg: float = 4.0,
    require_mod_w: float = 3.0,
    skip_with_lora: bool = True,
    fei_source: str = "z",
    run_dirs: list[Path] | None = None,
) -> list[Row]:
    """Walk bench output and gather per-(stem, seed) trajectory rows.

    ``fei_source`` selects how ``Row.fei_low`` gets populated:

    - ``"z"`` (default): paper-faithful 2-band FEI on the per-step latent
      ``z_t`` via ``library.runtime.fei.compute_fei_2band``. Read from the
      ``"fei_low"`` key in ``gaps_per_sample.npz`` if present, or from a
      ``fei_low.npz`` sidecar (written by
      ``scripts/dcw/collect_fei_sidecar.py``) for legacy pools whose main
      npz predates the FEI capture. Rows where neither is available stay
      ``Row.fei_low = None`` (trainer filter at ``--fei_obs != off``).

    - ``"v_surrogate"``: derive a 2-band simplex on the model output ``v_θ``
      from the per-sample Haar-band norms already in the main npz::

          e_low_v[r, i]  = v_rev_LL[r, i]² /
                          (v_rev_LL² + v_rev_LH² + v_rev_HL² + v_rev_HH²)[r, i]

      Zero-bench-cost first-look signal. Not paper-faithful (Haar-on-v ≠
      DoG-on-z) — captures the same coarse-vs-fine partition but on a
      different operand. Useful for deciding whether to invest in
      collecting real z-FEI on the rest of the pool.

    ``run_dirs`` (optional): explicit list of run dirs to load, bypassing
    the ``results_roots.iterdir()`` walk. Use for targeted training on a
    subset (single bucket / single experiment / replay-only test). When
    set, ``results_roots`` is ignored entirely.
    """
    if fei_source not in ("z", "v_surrogate"):
        raise ValueError(
            f"fei_source must be 'z' or 'v_surrogate', got {fei_source!r}"
        )
    if isinstance(results_roots, (str, Path)):
        results_roots = [Path(results_roots)]
    rows: list[Row] = []
    seen_run_names: set[str] = set()  # de-dup if same name appears in multiple roots
    candidate_dirs: list[Path] = []
    if run_dirs:
        # Explicit targeting — skip the root walk; consume the given dirs
        # in order. Caller is responsible for de-dup if they pass the same
        # path twice (we still hit the seen_run_names guard below).
        candidate_dirs.extend(Path(p) for p in run_dirs)
    else:
        for root in results_roots:
            if not root.exists():
                continue
            candidate_dirs.extend(p for p in root.iterdir() if p.is_dir())
    for run_dir in sorted(candidate_dirs):
        if run_dir.name in seen_run_names:
            continue
        seen_run_names.add(run_dir.name)
        npz_path = run_dir / "gaps_per_sample.npz"
        rj_path = run_dir / "result.json"
        if not (npz_path.exists() and rj_path.exists()):
            continue
        rj = json.loads(rj_path.read_text())
        a = rj.get("args", {})
        H, W = a.get("image_h"), a.get("image_w")
        if (H, W) not in DCW_ASPECT_TABLE:
            print(f"skip {run_dir.name}: aspect {H}x{W} not in table")
            continue
        if a.get("guidance_scale") != require_cfg:
            print(
                f"skip {run_dir.name}: cfg={a.get('guidance_scale')} != {require_cfg}"
            )
            continue
        if a.get("mod_w") != require_mod_w:
            print(f"skip {run_dir.name}: mod_w={a.get('mod_w')} != {require_mod_w}")
            continue
        if skip_with_lora and a.get("lora_weight"):
            print(f"skip {run_dir.name}: has LoRA {a['lora_weight']}")
            continue
        n_seeds = int(a.get("n_seeds", 1))
        z = np.load(npz_path, allow_pickle=True)
        stems = z["stems"]
        gap_LL = z["gap_LL"]  # (N, n_steps)
        if "v_rev_LL" in z.files:
            v_rev_LL = z["v_rev_LL"]
            source = "native"
        else:
            v_fwd_pop = _load_v_fwd_pop_mean(run_dir, band="LL")
            if v_fwd_pop is not None:
                v_rev_LL = (
                    gap_LL + v_fwd_pop[None, :]
                )  # broadcast (n_steps,) → (N, n_steps)
                source = "synthetic"
            else:
                v_rev_LL = gap_LL
                source = "fallback"
        sigma_i = _load_sigma_schedule(run_dir, n_steps=gap_LL.shape[1])
        aspect_id = DCW_ASPECT_TABLE[(H, W)]
        # Old runs predate --baseline_lambda; absent ⇒ 0.0 (legacy no-DCW).
        baseline_lambda = float(a.get("baseline_lambda", 0.0))
        # Per-row fei_low resolution order:
        #   1. v_surrogate mode: derive from existing v_rev_band norms.
        #   2. z mode + fei_low.npz sidecar present: read sidecar (rev-replay
        #      collector, scripts/dcw/collect_fei_sidecar.py).
        #   3. z mode + main npz has fei_low key: read it (post-capture runs).
        #   4. else: None (rows filtered out by trainer when --fei_obs != off).
        fei_low_arr: np.ndarray | None = None
        if fei_source == "v_surrogate":
            fei_low_arr = _derive_v_fei_surrogate(z)
        else:  # z mode
            sidecar = run_dir / "fei_low.npz"
            if sidecar.exists():
                sc = np.load(sidecar, allow_pickle=True)
                if "fei_low" in sc.files:
                    fei_low_arr = sc["fei_low"]
            if fei_low_arr is None and "fei_low" in z.files:
                fei_low_arr = z["fei_low"]
        for r in range(len(stems)):
            img_idx = r // n_seeds
            seed_idx = r % n_seeds
            rows.append(
                Row(
                    run_id=run_dir.name,
                    aspect_id=aspect_id,
                    stem=str(stems[r]),
                    seed_idx=int(
                        img_idx * 1000 + seed_idx
                    ),  # globally unique within run
                    gap_LL=np.asarray(gap_LL[r], dtype=np.float64),
                    v_rev_LL=np.asarray(v_rev_LL[r], dtype=np.float64),
                    v_rev_source=source,
                    sigma_i=sigma_i,
                    baseline_lambda=baseline_lambda,
                    fei_low=(
                        np.asarray(fei_low_arr[r], dtype=np.float64)
                        if fei_low_arr is not None
                        else None
                    ),
                )
            )
    return rows


_V_REV_BAND_KEYS = ("v_rev_LL", "v_rev_LH", "v_rev_HL", "v_rev_HH")


def _derive_v_fei_surrogate(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """2-band v-FEI surrogate from per-sample Haar bands of v_θ.

    Returns ``(N, n_steps)`` of e_low_v ∈ [0, 1] when all four bands are
    present; ``None`` otherwise (pre-band-capture runs). Squared band
    norms — same scale convention as Parseval-style energy. e_high is
    redundant for 2-band so we only emit e_low.
    """
    if not all(k in z.files for k in _V_REV_BAND_KEYS):
        return None
    ll = np.asarray(z["v_rev_LL"], dtype=np.float64)
    lh = np.asarray(z["v_rev_LH"], dtype=np.float64)
    hl = np.asarray(z["v_rev_HL"], dtype=np.float64)
    hh = np.asarray(z["v_rev_HH"], dtype=np.float64)
    total = ll**2 + lh**2 + hl**2 + hh**2
    # Avoid /0 at step boundaries where v is identically zero. The simplex
    # is undefined there; falling back to 0.5 leaves the head's input on
    # the simplex centroid (carries no information for that step).
    safe = np.where(total > 1e-12, total, 1.0)
    e_low = (ll**2) / safe
    e_low = np.where(total > 1e-12, e_low, 0.5)
    return e_low


def _load_v_fwd_pop_mean(run_dir: Path, *, band: str = "LL") -> np.ndarray | None:
    """Read baseline_v_fwd_<band> column from per_step_bands.csv as a per-step mean."""
    csv_path = run_dir / "per_step_bands.csv"
    if not csv_path.exists():
        return None
    col = f"baseline_v_fwd_{band}"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return np.array([float(r[col]) for r in reader], dtype=np.float64)
        except KeyError:
            return None


def _load_sigma_schedule(run_dir: Path, *, n_steps: int) -> np.ndarray:
    """Read the sigma_i column from per_step_bands.csv (or per_step.csv as fallback).

    The σ schedule is run-level (same across rows within a run), so we read
    once per run_dir. Falls back to a linear ramp from 1.0 → 0.0 over n_steps
    if neither csv carries the column — old runs without the schedule will
    silently degrade to that approximation.
    """
    for name in ("per_step_bands.csv", "per_step.csv"):
        csv_path = run_dir / name
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            try:
                arr = np.array([float(r["sigma_i"]) for r in reader], dtype=np.float64)
            except KeyError:
                continue
            if len(arr) >= n_steps:
                return arr[:n_steps]
    return np.linspace(1.0, 0.0, n_steps, dtype=np.float64)


def load_text_features(
    stems: list[str], dataset_dir: Path, variant: int = 0
) -> dict[str, dict]:
    """Per-stem c_pool + caption_length + token_l2_std from te cache."""
    out: dict[str, dict] = {}
    for stem in stems:
        if stem in out:
            continue
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            print(f"warn: missing te cache for {stem}")
            continue
        with safe_open(str(te_path), framework="pt") as f:
            emb = f.get_tensor(f"crossattn_emb_v{variant}").float()  # (512, 1024)
            mask = f.get_tensor(f"attn_mask_v{variant}").bool()  # (512,)
        valid = emb[mask]  # (L, 1024)
        if valid.numel() == 0:
            continue
        c_pool = valid.mean(dim=0)  # (1024,)
        token_l2 = valid.norm(dim=-1)  # (L,)
        out[stem] = {
            "c_pool": c_pool.numpy().astype(np.float32),
            "caption_length": int(mask.sum().item()),
            "token_l2_std": float(token_l2.std().item()),
        }
    return out


def build_population_mu_g(rows: list[Row], n_steps: int) -> np.ndarray:
    """Single population-mean LL gap trajectory across all rows."""
    if not rows:
        return np.zeros(n_steps, dtype=np.float64)
    return np.stack([r.gap_LL for r in rows]).mean(axis=0)
