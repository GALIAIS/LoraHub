"""Live preview image generation for diffusion-pipe runs.

dp doesn't render preview images during training — it only saves the LoRA
adapter every N steps. lorahub fills that gap with a background worker
that watches the job's output directory, runs an Anima-flavoured
inference for each new checkpoint, writes the result into the workspace
samples folder, and emits a `sample_ready` event so the UI picks up the
preview live.

Public surface:
    PreviewWorker  — owns the polling loop + inference dispatch
    PromptSpec     — one parsed line from the user's prompts_file
    parse_prompts_file — read & parse the prompts file
    AnimaInference — inference backend protocol (stub today, real tomorrow)

Design choices:
* The worker runs as a daemon thread inside the same uvicorn process the
  training subprocess was launched from. That keeps it tied to the job's
  lifetime — when the job ends or uvicorn dies, the worker dies with it.
* Inference happens *strictly between* checkpoints. We trigger only on
  the dp `Saving model to directory ...` line (which saver.py prints
  right after `save_adapter` returns), so the GPU has just released its
  microbatch buffers. We still hold a lock so a slow inference can't
  collide with the next checkpoint's inference.
* The first cut ships a stub backend that writes a deterministic
  placeholder PNG. The real Anima inference replaces only `AnimaInference`
  without touching the worker plumbing.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

from lorahub.core.events import EventType, TrainingEvent

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt parsing — sd-scripts compatible format.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PromptSpec:
    """One preview prompt line, post-parse.

    Mirrors the subset of sd-scripts' `line_to_prompt_dict` that makes
    sense for our preview pipeline: width / height / seed / steps / cfg
    scale / negative prompt. Anything else from the kohya format is
    intentionally dropped — preview images don't need ControlNet or
    img2img inputs.
    """

    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    negative: str | None = None
    # Original line index (for stable filenames).
    index: int = 0


_FLAG_RE = re.compile(r"\s+--\s*([a-zA-Z]+)\s+(.*?)(?=\s+--\s*[a-zA-Z]+\s|$)")


def parse_prompts_file(path: Path) -> list[PromptSpec]:
    """Read a kohya-style prompts file. Blank lines and `#` comments
    are skipped; everything else becomes a PromptSpec."""
    if not path.is_file():
        return []
    out: list[PromptSpec] = []
    raw = path.read_text(encoding="utf-8")
    for i, raw_line in enumerate(raw.splitlines()):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(_parse_line(stripped, index=len(out)))
    return out


def _parse_line(line: str, *, index: int) -> PromptSpec:
    # Find the first ` --x ` flag boundary; everything before is the prompt body.
    first = re.search(r"\s+--[a-zA-Z]+\s", line)
    body = line[: first.start()] if first else line
    spec = PromptSpec(prompt=body.strip(), index=index)
    if not first:
        return spec
    tail = line[first.start() :]
    for m in _FLAG_RE.finditer(tail):
        key = m.group(1).lower()
        value = m.group(2).strip()
        try:
            if key == "w":
                spec.width = int(value)
            elif key == "h":
                spec.height = int(value)
            elif key == "d":
                spec.seed = int(value)
            elif key == "s":
                spec.steps = max(1, min(1000, int(value)))
            elif key == "l":
                spec.cfg = float(value)
            elif key == "n":
                spec.negative = value
            # `g`/`ss`/`cn`/`i`/etc. — dropped; the preview pipeline
            # doesn't model conditioning beyond the basics.
        except (ValueError, TypeError):
            log.warning("preview prompt %d: bad --%s %r", index, key, value)
    return spec


# --------------------------------------------------------------------------- #
# Inference backend protocol + stub implementation
# --------------------------------------------------------------------------- #


class AnimaInference(Protocol):
    """Minimum contract for an inference backend.

    The real implementation will load Anima base + Qwen3 TE + VAE once
    and reuse them across calls. The stub bypasses all of that and just
    paints a deterministic PNG so the rest of the pipeline can be
    validated end-to-end without GPU.
    """

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None: ...


class StubInference:
    """Placeholder inference. Writes a deterministic PNG so the worker
    + event flow can be wired and exercised before the real Anima
    inference lands."""

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
        except ImportError:
            # Fall back to a 1-byte placeholder rather than crashing the
            # worker. The UI shows a broken-image marker which is still
            # informative ("the worker tried but Pillow isn't installed").
            out_path.write_bytes(b"")
            return

        # Hash of (lora path mtime + prompt + seed) -> deterministic colour
        # so different checkpoints render visibly different blocks.
        mtime = int(lora_path.stat().st_mtime) if lora_path.exists() else 0
        seed = spec.seed if spec.seed is not None else 42
        h = hash((mtime, spec.prompt, seed)) & 0xFFFFFF
        r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
        img = Image.new("RGB", (spec.width, spec.height), (r, g, b))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None
        text = (
            f"PREVIEW STUB\n"
            f"prompt[{spec.index}]: {spec.prompt[:60]}\n"
            f"size: {spec.width}x{spec.height}\n"
            f"seed: {seed} steps: {spec.steps or default_steps} cfg: {spec.cfg or default_cfg}\n"
            f"lora: {lora_path.name}"
        )
        draw.multiline_text((24, 24), text, fill="white", font=font, spacing=6)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG", optimize=True)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PreviewConfig:
    enabled: bool
    prompts_file: Path | None
    default_steps: int
    default_cfg: float
    samples_dir: Path
    output_dir: Path
    poll_interval_s: float = 5.0
    # Hard cap on time spent rendering a single checkpoint's prompts.
    # Even with a 4-image prompt set we shouldn't burn more than this
    # much wall time per ckpt — once exceeded, the worker breaks out
    # of the prompt loop, drops the rest of that checkpoint's prompts
    # on the floor, and waits for the next one. Rationale: lazy
    # protection so a degenerate inference call can't permanently
    # starve training throughput.
    max_render_time_per_ckpt_s: float = 300.0
    # Soft budget — a fraction of the wall-clock distance between the
    # last two checkpoints. A checkpoint cadence of 200 steps × 3.7s
    # = 740s; with budget_fraction=0.3 that's 222s of preview work
    # allowed. Effective budget = min(max_render_time, fraction × Δ).
    # Falls back to max_render_time on the first ckpt (no Δ yet).
    budget_fraction: float = 0.3


@dataclass(slots=True)
class _CheckpointSeen:
    name: str
    mtime: float
    rendered: bool = False


class PreviewWorker:
    """Background loop that turns new dp checkpoints into preview PNGs.

    The worker is single-threaded and serial: even when several prompts
    are in the file, each is rendered one after the other and only after
    the full batch finishes do we move on to the next checkpoint. This
    matches our GPU sharing strategy — never run inference in parallel
    with another inference, and never run two checkpoints' worth of
    previews at once.

    Trigger model:
      * Primary: explicit `notify_checkpoint(ckpt_name)` calls from
        whoever forwards `checkpoint_saved` events. These give us the
        sub-second reaction we want — the moment dp's saver returns,
        the worker is already rendering against the freshly written
        adapter.
      * Fallback: a polling tick that scans `output_dir/{step,epoch}*`
        directories. Catches checkpoints that arrived while the worker
        was busy and any case where the event channel was lossy.

    Both paths converge on the same `_seen` map keyed by ckpt name +
    mtime so we can never render the same checkpoint twice.
    """

    def __init__(
        self,
        *,
        config: PreviewConfig,
        inference: AnimaInference,
        on_event: Callable[[TrainingEvent], None],
        job_id: str,
        stop_evt: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.inference = inference
        self.on_event = on_event
        self.job_id = job_id
        self.stop_evt = stop_evt or threading.Event()
        self._seen: dict[str, _CheckpointSeen] = {}
        self._notify_q: queue.Queue[str] = queue.Queue()
        # Wall-clock at the last successful checkpoint render — used to
        # compute the per-checkpoint budget.
        self._last_render_completed_at: float | None = None
        self._last_render_started_at: float | None = None

    def notify_checkpoint(self, ckpt_name: str) -> None:
        """Wake the worker on a `checkpoint_saved` event so it doesn't
        have to wait for the next polling tick. Safe to call from any
        thread; falls through harmlessly if the worker isn't running."""
        try:
            self._notify_q.put_nowait(ckpt_name)
        except queue.Full:
            # Bounded fallback: if somehow the queue fills (shouldn't —
            # we don't bound it), drop the notify and rely on polling.
            pass

    def run(self) -> None:
        if not self.config.enabled:
            return
        prompts = (
            parse_prompts_file(self.config.prompts_file)
            if self.config.prompts_file is not None
            else []
        )
        if not prompts:
            log.info(
                "preview worker [%s]: no prompts (file=%s) — skipping",
                self.job_id,
                self.config.prompts_file,
            )
            return
        log.info(
            "preview worker [%s]: started, %d prompts, watching %s",
            self.job_id,
            len(prompts),
            self.config.output_dir,
        )

        while not self.stop_evt.is_set():
            # Wait either for an event-driven notify or for the polling
            # tick — whichever fires first.
            try:
                self._notify_q.get(timeout=self.config.poll_interval_s)
            except queue.Empty:
                pass
            if self.stop_evt.is_set():
                break
            # Drain any extra notifies that piled up while we were
            # blocking — _tick will pick the latest state from disk
            # anyway, no point in re-running for each one.
            with _drain_queue(self._notify_q):
                pass
            try:
                self._tick(prompts)
            except Exception:  # noqa: BLE001
                log.exception("preview worker [%s] tick failed", self.job_id)

    def _tick(self, prompts: list[PromptSpec]) -> None:
        out_root = self.config.output_dir
        if not out_root.is_dir():
            return
        for ckpt_dir in sorted(_iter_ckpt_dirs(out_root)):
            adapter = _adapter_in(ckpt_dir)
            if adapter is None:
                continue
            mtime = adapter.stat().st_mtime
            entry = self._seen.get(ckpt_dir.name)
            if entry is None:
                entry = _CheckpointSeen(name=ckpt_dir.name, mtime=mtime)
                self._seen[ckpt_dir.name] = entry
            elif entry.rendered and entry.mtime == mtime:
                continue
            elif entry.mtime != mtime:
                # Adapter file was rewritten — re-render.
                entry.mtime = mtime
                entry.rendered = False

            self._render_one(adapter, ckpt_dir.name, prompts)
            entry.rendered = True

    def _render_one(
        self, adapter: Path, ckpt_name: str, prompts: list[PromptSpec]
    ) -> None:
        budget = self._compute_budget()
        ckpt_started = time.time()
        self._last_render_started_at = ckpt_started
        log.info(
            "preview worker [%s] rendering %s (%d prompts, budget=%.0fs)",
            self.job_id,
            ckpt_name,
            len(prompts),
            budget,
        )
        budget_exceeded = False
        for spec in prompts:
            if self.stop_evt.is_set():
                return
            elapsed = time.time() - ckpt_started
            if elapsed >= budget:
                budget_exceeded = True
                break
            png_name = f"{ckpt_name}_{spec.index:02d}.png"
            out_path = self.config.samples_dir / png_name
            try:
                started = time.time()
                self.inference.render(
                    lora_path=adapter,
                    spec=spec,
                    out_path=out_path,
                    default_steps=self.config.default_steps,
                    default_cfg=self.config.default_cfg,
                )
                duration = time.time() - started
            except Exception as exc:  # noqa: BLE001
                if _is_skipped(exc):
                    # Backend deliberately bowed out (low VRAM, cancelled).
                    # Nothing to log loudly — the next ckpt picks up.
                    log.info(
                        "preview worker [%s] %s prompt %d skipped: %s",
                        self.job_id,
                        ckpt_name,
                        spec.index,
                        exc,
                    )
                    continue
                log.exception(
                    "preview worker [%s] render failed for %s prompt %d",
                    self.job_id,
                    ckpt_name,
                    spec.index,
                )
                self._emit_log(
                    "error",
                    f"preview render failed for {ckpt_name} prompt {spec.index}: {exc}",
                )
                continue
            log.info(
                "preview worker [%s] rendered %s in %.1fs",
                self.job_id,
                png_name,
                duration,
            )
            self.on_event(
                TrainingEvent(
                    type=EventType.sample_ready,
                    job_id=self.job_id,
                    payload={
                        "path": str(out_path),
                        "checkpoint": ckpt_name,
                        "prompt_index": spec.index,
                        "duration_s": round(duration, 2),
                    },
                )
            )
        if budget_exceeded:
            remaining = max(0, len(prompts) - spec.index - 1)
            log.warning(
                "preview worker [%s] budget exceeded on %s after %.1fs — "
                "dropping %d remaining prompt(s)",
                self.job_id,
                ckpt_name,
                time.time() - ckpt_started,
                remaining,
            )
            self._emit_log(
                "warning",
                f"preview budget exceeded on {ckpt_name}; dropped "
                f"{remaining} prompt(s) to keep training throughput up",
            )
        self._last_render_completed_at = time.time()

    def _compute_budget(self) -> float:
        """Effective per-checkpoint render budget in seconds."""
        cap = max(0.0, self.config.max_render_time_per_ckpt_s)
        if self._last_render_completed_at is None:
            return cap
        elapsed_since = time.time() - self._last_render_completed_at
        soft = max(0.0, elapsed_since * self.config.budget_fraction)
        return min(cap, soft)

    def _emit_log(self, level: str, message: str) -> None:
        try:
            self.on_event(
                TrainingEvent(
                    type=EventType.log,
                    job_id=self.job_id,
                    payload={"level": level, "message": message, "source": "preview"},
                )
            )
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_skipped(exc: BaseException) -> bool:
    """True for exceptions whose class name ends in `Skipped` — currently
    `InferenceSkipped` from `lorahub.core.inference.anima`. This is
    duck-typed on the class name to avoid an import cycle (the anima
    backend imports from this module)."""
    return type(exc).__name__.endswith("Skipped")


import contextlib


@contextlib.contextmanager
def _drain_queue(q: "queue.Queue[str]"):
    """Drop any pending items from `q` without blocking. Used after a
    notify to coalesce a burst of `checkpoint_saved` signals into one
    `_tick` call (which scans the whole output dir anyway)."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
    yield


def _iter_ckpt_dirs(output_dir: Path) -> Iterable[Path]:
    """Yield dp checkpoint dirs (`step{N}` / `epoch{N}`).

    diffusion-pipe's ``train.py`` always prepends a UTC timestamp run-dir
    under the configured ``output_dir`` (see ``get_most_recent_run_dir``
    upstream), so the real layout is ``<output_dir>/<YYYYMMDD_HH-MM-SS>/
    step{N}/`` rather than ``<output_dir>/step{N}/``. We mirror dp's
    selection logic — pick the alphabetically-last subdirectory as the
    active run — and fall back to scanning ``output_dir`` directly so
    older / hand-laid layouts still work.
    """
    if not output_dir.is_dir():
        return
    # Direct layout: output_dir/step* | epoch*
    direct = [
        p
        for p in output_dir.iterdir()
        if p.is_dir() and (p.name.startswith("step") or p.name.startswith("epoch"))
    ]
    if direct:
        yield from direct
        return
    # Nested layout: output_dir/<run>/step* | epoch*. dp picks the
    # alphabetically-last child via `sorted(...)[-1]`; we replicate that
    # so a fresh run picks up its own ckpts (timestamps sort lexicographically).
    candidates = sorted(p for p in output_dir.iterdir() if p.is_dir())
    if not candidates:
        return
    run_dir = candidates[-1]
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("step") or child.name.startswith("epoch"):
            yield child


def _adapter_in(ckpt_dir: Path) -> Path | None:
    """Pick the adapter weights file in `ckpt_dir`, if present.

    dp's saver writes a single `*.safetensors` per ckpt dir (the LoRA
    adapter merged into one tensor file). We don't need to know the
    exact name — there's only one safetensors at this layer.
    """
    candidates = sorted(ckpt_dir.glob("*.safetensors"))
    return candidates[0] if candidates else None
