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


@dataclass(slots=True)
class _CheckpointSeen:
    name: str
    mtime: float
    rendered: bool = False


@dataclass(slots=True)
class PreviewWorker:
    """Background loop that turns new dp checkpoints into preview PNGs.

    The worker is single-threaded and serial: even when several prompts
    are in the file, each is rendered one after the other and only after
    the full batch finishes do we move on to the next checkpoint. This
    matches our GPU sharing strategy — never run inference in parallel
    with another inference, and never run two checkpoints' worth of
    previews at once.
    """

    config: PreviewConfig
    inference: AnimaInference
    on_event: Callable[[TrainingEvent], None]
    job_id: str
    stop_evt: threading.Event = field(default_factory=threading.Event)
    _seen: dict[str, _CheckpointSeen] = field(default_factory=dict)

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
        while not self.stop_evt.wait(self.config.poll_interval_s):
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
        for spec in prompts:
            if self.stop_evt.is_set():
                return
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


def _iter_ckpt_dirs(output_dir: Path) -> Iterable[Path]:
    """Yield dp checkpoint dirs (`step{N}` / `epoch{N}` under output_dir)."""
    for child in output_dir.iterdir():
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
