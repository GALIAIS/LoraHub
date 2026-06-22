"""AnimaLoraBackend: wraps the vendored sorryhyun/anima_lora as a TrainingBackend.

Translates a ``TrainingConfig`` into the override-layer CLI argv anima_lora
expects (see ``compiler.py``), and launches ``<python> -m
accelerate.commands.accelerate_cli launch train.py <args>`` through the
shared ``SubprocessRunner`` (see ``runner.py``).

We deliberately keep the supported arch set narrow: anima_lora's reason
to exist is its Anima-specific algorithm stack (OrthoLoRA / T-LoRA /
Hydra / postfix / EasyControl / IP-Adapter). Recipes targeting
non-anima archs land in the kohya / dp backends instead.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import ulid

from lorahub.core.backends._common.vram import estimate_vram as _shared_estimate_vram
from lorahub.core.backends.anima_lora import bootstrap as _bootstrap
from lorahub.core.backends.anima_lora.compiler import (
    DEFAULT_SAMPLE_PROMPTS_FILENAME,
    CompilationError,
    compile_config,
    compile_turbo_config,
)
from lorahub.core.backends.anima_lora.preprocess import (
    PreprocessError,
    ensure_cache,
)
from lorahub.core.backends.anima_lora.runner import AnimaLoraRunner
from lorahub.core.backends.anima_lora.turbo_runner import AnimaLoraTurboRunner
from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent

# anima_lora is purpose-built for Anima DiT; everything else falls
# through to kohya / dp. Keeping this set tight means the validator
# can give a clear error when a user accidentally points the wrong
# arch at this backend.
_SUPPORTED: set[ModelArch] = {ModelArch.anima}


class AnimaLoraBackend:
    """Wraps the vendored sorryhyun/anima_lora source as a TrainingBackend."""

    @property
    def name(self) -> str:
        return "anima_lora"

    @property
    def supported_archs(self) -> set[ModelArch]:
        return set(_SUPPORTED)

    def validate(self, cfg: TrainingConfig) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if cfg.base_model.arch not in {a.value for a in _SUPPORTED}:
            issues.append(
                ValidationIssue(
                    Severity.error,
                    "base_model.arch",
                    (
                        f"anima_lora does not support arch {cfg.base_model.arch!r}; "
                        f"supported: {sorted(a.value for a in _SUPPORTED)}. "
                        "Switch backend.type to 'kohya' or 'diffusion-pipe' "
                        "for other arches."
                    ),
                )
            )

        # Bootstrap probe against the vendored copy. The repo_path field
        # is reused from BackendConfig; users normally leave it None and
        # we resolve to external/anima_lora/ automatically.
        try:
            _bootstrap.resolve(
                config_path=cfg.backend.repo_path,
                config_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as e:
            issues.append(
                ValidationIssue(Severity.error, "backend.repo_path", str(e))
            )

        # Pass an arbitrary workspace — compile_config doesn't write
        # files (anima_lora owns its own merge chain) so the path is
        # only used to construct --output_dir.
        try:
            compile_config(cfg, workspace=Path("/"))
        except CompilationError as e:
            issues.append(ValidationIssue(Severity.error, "backend.animaLora", str(e)))

        # Cross-field consistency rules — torch.compile vs offload, EMA vs
        # cudagraph_trees, 8bit-optimizer requires bnb, network/alpha ratio,
        # validation_split_num vs useCmmd, etc. See policies.py for the
        # full catalogue and rationale per rule.
        from lorahub.core.backends.anima_lora.policies import (  # noqa: PLC0415
            check_cross_field_conflicts,
        )

        issues.extend(check_cross_field_conflicts(cfg))

        if not cfg.base_model.checkpoint.exists():
            issues.append(
                ValidationIssue(
                    Severity.warning,
                    "base_model.checkpoint",
                    f"checkpoint file does not exist: {cfg.base_model.checkpoint}",
                )
            )
        if not cfg.dataset.source.exists():
            issues.append(
                ValidationIssue(
                    Severity.warning,
                    "dataset.source",
                    f"dataset directory does not exist: {cfg.dataset.source}",
                )
            )

        return issues

    def estimate_vram(self, cfg: TrainingConfig) -> VRAMEstimate:
        """Reuses the shared `_common.vram` anima entry.

        Upstream reports 13.4 GB peak at rank=32, 1MP on a 5060 Ti, but
        the shared estimator is conservative and works off the
        precision / batch_size / rank knobs we already track. Good
        enough for the UI to flag ``estimated > available_vram`` cases
        before launch; a tighter model can land later if needed.
        """
        return _shared_estimate_vram(
            cfg.base_model.arch,
            precision=cfg.precision,
            batch_size=cfg.schedule.batch_size,
            network_rank=cfg.network.rank,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

    def launch(
        self,
        cfg: TrainingConfig,
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
        *,
        extra_argv: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> TrainingHandle:
        bootstrap_env = _bootstrap.resolve(
            config_path=cfg.backend.repo_path,
            config_python=cfg.backend.python_executable,
        )
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        # Wandb seed: anima's pyproject doesn't depend on wandb, but the
        # compiler emits --log_with=wandb whenever monitoring is enabled.
        # Install on demand so the user doesn't hit ImportError mid-train
        # after caching has already burned ~minutes of GPU time.
        _ensure_wandb_if_enabled(cfg, bootstrap_env.python_executable, on_event)

        # Caption sanitisation — shared with kohya / dp; see
        # _common.dataset_prep.
        from lorahub.core.backends._common.dataset_prep import (  # noqa: PLC0415
            apply_caption_dropouts,
        )
        apply_caption_dropouts(cfg, workspace)

        # Auto-preprocess: ensure the LoRA cache under
        # <workspace>/post_image_dataset/lora is populated before the
        # trainer reads it. This keeps cfg.dataset.source pointing at
        # the user's raw image dir (same shape kohya / dp use) instead
        # of forcing them to ``make preprocess`` separately. Failures
        # turn into a CompilationError so the launcher returns a clear
        # error rather than a half-running training subprocess.
        try:
            ensure_cache(
                image_dir=cfg.dataset.source,
                workspace=workspace,
                base_model=cfg.base_model,
                env=bootstrap_env,
                on_event=on_event,
                opts=cfg.backend.anima_lora,
            )
        except PreprocessError as e:
            on_event(
                TrainingEvent(
                    type=EventType.error,
                    payload={"source": "preprocess", "error": str(e)},
                )
            )
            msg = f"anima_lora auto-preprocess failed: {e}"
            raise CompilationError(msg) from e

        _prepare_sample_prompts_file(cfg, workspace)

        # Branch: turbo distillation (scripts/distill_turbo.py) vs the
        # regular train.py path. Turbo is picked when the recipe has
        # backend.animaLora.turbo populated; both paths share workspace
        # setup but diverge on argv shape + runner choice.
        opts = cfg.backend.anima_lora
        is_turbo = opts is not None and opts.turbo is not None
        if is_turbo:
            argv, files = compile_turbo_config(cfg, workspace)
        else:
            argv, files = compile_config(cfg, workspace)
        if extra_argv:
            argv = [*argv, *extra_argv]
        # `files` is always empty for anima_lora — kept for shape parity
        # with kohya / dp launchers.
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        job_id = str(ulid.new())
        runner: AnimaLoraRunner | AnimaLoraTurboRunner
        if is_turbo:
            runner = AnimaLoraTurboRunner(
                python=bootstrap_env.python_executable,
                repo=bootstrap_env.repo_path,
                argv=argv,
                workspace=workspace,
                on_event=on_event,
                job_id=job_id,
                env=env,
            )
        else:
            runner = AnimaLoraRunner(
                python=bootstrap_env.python_executable,
                repo=bootstrap_env.repo_path,
                argv=argv,
                workspace=workspace,
                on_event=on_event,
                job_id=job_id,
                env=env,
            )
        runner.start()

        # Anima writes sample PNGs to ``<output_dir>/sample/`` directly
        # via PIL.image.save() — no stdout chatter, so the parser can't
        # detect them. Start a tiny watcher thread that polls the
        # directory and emits a ``sample_ready`` event for every new
        # file. It exits automatically once the trainer subprocess
        # terminates.
        if cfg.sampling.enabled:
            sample_dir = workspace / "ckpt" / "sample"
            _start_sample_watcher(
                sample_dir=sample_dir,
                workspace=workspace,
                on_event=on_event,
                runner=runner,
                job_id=job_id,
            )

        return TrainingHandle(
            job_id=job_id,
            pid=runner.pid,
            _stop_fn=lambda graceful: runner.stop(graceful=graceful),
            _wait_fn=lambda timeout: runner.wait(timeout=timeout).returncode,
        )


__all__ = ["AnimaLoraBackend"]


# --- Sample directory watcher ---------------------------------------------
# anima emits PNGs straight to disk via PIL without any stdout marker, so
# the SubprocessRunner's stdout-pumping parser can't see them go by. A
# polling watcher closes the gap: scandir the sample directory every few
# seconds, diff against the previous snapshot, and emit ``sample_ready``
# for each fresh file. The thread terminates when the trainer process
# exits, so there's no separate stop signal to plumb through.

# Polling cadence. anima's sampler is bursty (one image per prompt every
# ``sample_every_n_epochs``), so a slow tick is fine — we just want to
# beat the user's "is it done yet" attention span.
_SAMPLE_POLL_INTERVAL = 3.0
# Stop watching ``_SAMPLE_GRACE_AFTER_EXIT`` seconds after the trainer
# subprocess has exited. anima sometimes writes a final batch right at
# the end-of-training save_state hook; without a small grace window we
# would miss those PNGs.
_SAMPLE_GRACE_AFTER_EXIT = 8.0
_SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _start_sample_watcher(
    *,
    sample_dir: Path,
    workspace: Path,
    on_event: Callable[[TrainingEvent], None],
    runner: AnimaLoraRunner | AnimaLoraTurboRunner,
    job_id: str,
) -> None:
    seen: set[str] = set()

    def watch() -> None:
        # Don't probe ``runner._proc`` — the SubprocessRunner private API
        # may evolve. Instead poll until the subprocess has been gone for
        # a grace window. ``runner.pid`` is None until ``start()`` has
        # taken hold; ``_pid_alive`` is the same probe the orphan reaper
        # uses, so behaviour stays consistent across the codebase.
        from lorahub.api.store import _pid_alive  # noqa: PLC0415

        exit_seen_at: float | None = None
        while True:
            try:
                if sample_dir.is_dir():
                    for path in sample_dir.iterdir():
                        if path.suffix.lower() not in _SAMPLE_SUFFIXES:
                            continue
                        if not path.is_file():
                            continue
                        rel = path.relative_to(workspace).as_posix()
                        if rel in seen:
                            continue
                        seen.add(rel)
                        try:
                            mtime = path.stat().st_mtime
                            size = path.stat().st_size
                        except OSError:
                            continue
                        on_event(
                            TrainingEvent(
                                type=EventType.sample_ready,
                                payload={
                                    "path": rel,
                                    "size_bytes": size,
                                    "modified_at": mtime,
                                    "filename": path.name,
                                },
                                job_id=job_id,
                            )
                        )
            except Exception:  # noqa: BLE001 — never let the watcher kill the job
                pass

            pid = runner.pid
            alive = pid is not None and _pid_alive(pid)
            if not alive:
                if exit_seen_at is None:
                    exit_seen_at = time.time()
                elif time.time() - exit_seen_at > _SAMPLE_GRACE_AFTER_EXIT:
                    return
            else:
                exit_seen_at = None
            time.sleep(_SAMPLE_POLL_INTERVAL)

    thread = threading.Thread(
        target=watch,
        name=f"anima-samples-{job_id[-6:]}",
        daemon=True,
    )
    thread.start()


# --- Sample prompts fallback -----------------------------------------------
# anima train.py refuses to do mid-run sampling without ``--sample_prompts
# <file>``. When the user enables sampling in the recipe but doesn't point
# at a custom prompts file, we synthesise one from the training dataset's
# own captions. The result is intentionally simple — a plain `.txt` with
# one prompt per line — because anima parses that format directly. PNGs
# land under ``<output_dir>/sample/`` and LoraHub's sample router picks
# them up via rglob automatically.
_FALLBACK_PROMPT = "a high quality detailed illustration"
# Cap so we don't spend a chunk of every epoch on samples — anima
# generates one image per prompt per cadence tick.
_MAX_FALLBACK_PROMPTS = 3
_ANIMA_SAMPLE_SAFE_PROMPTS_FILENAME = "_lorahub_anima_sample_prompts.txt"
_PROMPT_DIM_FLAG_RE = re.compile(r"(?<!\S)--(?P<flag>[wh])\s+(?P<value>\d+)(?!\S)")


def _gather_dataset_captions(source: Path, limit: int) -> list[str]:
    """Pick up to ``limit`` non-empty caption strings from ``source``.

    Reads the first ``.txt`` sidecars sorted lexicographically so the
    result is deterministic across re-launches (matters for reproducible
    sample image diffs between epochs). Empty / missing files are
    silently skipped; if nothing usable is found the caller falls back
    to a generic prompt.
    """
    if not source.is_dir():
        return []
    out: list[str] = []
    for path in sorted(source.rglob("*.txt")):
        if len(out) >= limit:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        # Anima treats one line == one prompt. Fold multi-line caption
        # files (rare but possible from tagger pipelines) onto a single
        # line so we don't accidentally split one image's caption into
        # two prompts.
        text = " ".join(text.split())
        out.append(text)
    return out


def _prepare_sample_prompts_file(cfg: TrainingConfig, workspace: Path) -> None:
    """Ensure anima sampling uses dimensions compatible with static padding.

    The DiT sample path reuses the training model. In legacy
    ``static_token_count`` mode, sample image patches must fit the same
    ``(width // 16) * (height // 16)`` token budget as the compiled
    training buckets. Over-budget prompt sizes fail later with an opaque
    ``unflatten`` shape error, so LoraHub clamps only preview dimensions
    before launching train.py.
    """
    sampling = cfg.sampling
    if not sampling.enabled:
        return
    if sampling.prompts_file is not None:
        _sanitize_existing_sample_prompts_file(cfg, workspace)
        return
    _ensure_sample_prompts_file(cfg, workspace)


def _ensure_sample_prompts_file(cfg: TrainingConfig, workspace: Path) -> None:
    """Write a fallback prompts file under the workspace if needed."""
    sampling = cfg.sampling
    target = workspace / DEFAULT_SAMPLE_PROMPTS_FILENAME

    captions = _gather_dataset_captions(
        Path(str(cfg.dataset.source)), _MAX_FALLBACK_PROMPTS
    )
    if not captions:
        captions = [_FALLBACK_PROMPT]

    width = sampling.resolution[0] if sampling.resolution else 1024
    height = sampling.resolution[1] if len(sampling.resolution) > 1 else width
    width, height = _clamp_sample_dimensions(cfg, int(width), int(height))
    seed_part = ""
    # ``sampling.seed`` is the *training* seed; ``-1`` is the legacy
    # ComfyUI-style "randomise at run time" sentinel that the lifecycle
    # hook resolves to a concrete integer before launch. By the time we
    # land here it should never be ``-1``, but if it ever does we omit
    # ``--d`` so anima train.py's ``_sample_image_inference`` falls back
    # to the ambient RNG (fresh noise per epoch) instead of pinning every
    # epoch to the literal ``-1`` and producing identical previews.
    if int(sampling.seed) >= 0:
        seed_part = f" --d {int(sampling.seed)}"
    suffix = (
        f" --w {int(width)} --h {int(height)}"
        f"{seed_part}"
        f" --s {int(sampling.inference_steps)}"
        f" --l {sampling.inference_cfg}"
    )
    body = "\n".join(prompt + suffix for prompt in captions) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _sanitize_existing_sample_prompts_file(
    cfg: TrainingConfig,
    workspace: Path,
) -> None:
    source = Path(str(cfg.sampling.prompts_file))
    if not source.is_file() or source.suffix.lower() != ".txt":
        return
    target = workspace / _ANIMA_SAMPLE_SAFE_PROMPTS_FILENAME
    changed = False
    safe_lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line, line_changed = _clamp_prompt_line_dimensions(cfg, raw_line)
        safe_lines.append(line)
        changed = changed or line_changed
    if not changed:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(safe_lines) + "\n", encoding="utf-8")
    cfg.sampling.prompts_file = target


def _clamp_prompt_line_dimensions(
    cfg: TrainingConfig,
    line: str,
) -> tuple[str, bool]:
    flags: dict[str, int] = {}
    for match in _PROMPT_DIM_FLAG_RE.finditer(line):
        flags[match.group("flag").lower()] = int(match.group("value"))
    if "w" not in flags and "h" not in flags:
        return line, False

    fallback_width = (
        cfg.sampling.resolution[0] if cfg.sampling.resolution else 1024
    )
    fallback_height = (
        cfg.sampling.resolution[1]
        if len(cfg.sampling.resolution) > 1
        else fallback_width
    )
    width = flags.get("w", int(fallback_width))
    height = flags.get("h", int(fallback_height))
    safe_width, safe_height = _clamp_sample_dimensions(cfg, width, height)
    if safe_width == width and safe_height == height:
        return line, False

    found = {"w": False, "h": False}

    def replace(match: re.Match[str]) -> str:
        flag = match.group("flag").lower()
        found[flag] = True
        value = safe_width if flag == "w" else safe_height
        return f"--{flag} {value}"

    updated = _PROMPT_DIM_FLAG_RE.sub(replace, line)
    if not found["w"]:
        updated = f"{updated} --w {safe_width}"
    if not found["h"]:
        updated = f"{updated} --h {safe_height}"
    return updated, True


def _clamp_sample_dimensions(
    cfg: TrainingConfig,
    width: int,
    height: int,
) -> tuple[int, int]:
    opts = cfg.backend.anima_lora
    token_budget = None
    if opts is not None and not opts.enable_native_flatten:
        token_budget = opts.static_token_count
    if token_budget is None:
        return (
            _align_sample_dimension(width),
            _align_sample_dimension(height),
        )

    width = _align_sample_dimension(width)
    height = _align_sample_dimension(height)
    if _sample_token_count(width, height) <= token_budget:
        return width, height

    scale = math.sqrt(token_budget / _sample_token_count(width, height))
    safe_width = _align_sample_dimension(math.floor(width * scale), mode="floor")
    safe_height = _align_sample_dimension(math.floor(height * scale), mode="floor")
    while _sample_token_count(safe_width, safe_height) > token_budget:
        if safe_width >= safe_height and safe_width > 64:
            safe_width -= 16
        elif safe_height > 64:
            safe_height -= 16
        else:
            break
    return _fill_sample_token_budget(
        safe_width,
        safe_height,
        token_budget=token_budget,
        target_ratio=width / height,
    )


def _fill_sample_token_budget(
    width: int,
    height: int,
    *,
    token_budget: int,
    target_ratio: float,
) -> tuple[int, int]:
    while True:
        candidates = [
            (width + 16, height),
            (width, height + 16),
        ]
        candidates = [
            candidate
            for candidate in candidates
            if _sample_token_count(*candidate) <= token_budget
        ]
        if not candidates:
            return width, height
        width, height = min(
            candidates,
            key=lambda candidate: (
                abs(math.log((candidate[0] / candidate[1]) / target_ratio)),
                -_sample_token_count(*candidate),
            ),
        )


def _align_sample_dimension(value: int | float, *, mode: str = "nearest") -> int:
    value = max(64, int(value))
    if mode == "floor":
        return max(64, value - value % 16)
    return max(64, value - value % 16)


def _sample_token_count(width: int, height: int) -> int:
    return max(1, width // 16) * max(1, height // 16)


def _ensure_wandb_if_enabled(
    cfg: TrainingConfig,
    venv_python: Path,
    on_event: Callable[[TrainingEvent], None],
) -> None:
    """Lazily install wandb into the anima venv when monitoring is on.

    anima_lora's ``pyproject.toml`` only declares ``tensorboard`` — wandb
    isn't pulled in by ``uv sync``. But ``library/runtime/accelerator.py``
    hard-imports wandb whenever ``--log_with wandb`` is set, which the
    compiler emits whenever ``cfg.monitoring.enable_wandb=true``. Without
    a probe step the user only finds out after caching is done, mid-way
    through ``prepare_accelerator`` — and at that point a long preprocess
    run has been wasted on a config problem we could have detected in
    ~200 ms before the spawn.

    Probe is a one-shot ``python -c "import wandb"``; install is
    ``uv pip install --python <venv> wandb`` (preferred — respects
    anima's index pins) with a plain ``pip`` fallback. Idempotent on
    success; raises ``CompilationError`` with an actionable message
    when both probe and install fail so the user can fix it without
    digging through the traceback.
    """
    if not cfg.monitoring.enable_wandb:
        return

    import subprocess  # noqa: PLC0415

    def _probe() -> tuple[int, str]:
        proc = subprocess.run(  # noqa: S603
            [str(venv_python), "-c", "import wandb"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode, (proc.stderr or proc.stdout or "").strip()

    rc, _ = _probe()
    if rc == 0:
        return

    on_event(
        TrainingEvent(
            type=EventType.log,
            payload={
                "source": "wandb-install",
                "message": "anima venv 缺少 wandb,monitoring.enableWandb=true,正在安装……",
            },
        )
    )

    # Prefer uv (fast, respects anima's [tool.uv.sources] pins). Fall
    # back to plain pip so the path still works if uv isn't on PATH.
    try:
        from lorahub.core.toolchain.uv import find_uv  # noqa: PLC0415
        uv = find_uv()
    except Exception:  # noqa: BLE001
        uv = None
    if uv:
        install_cmd = [uv, "pip", "install", "--python", str(venv_python), "wandb"]
    else:
        install_cmd = [str(venv_python), "-m", "pip", "install", "wandb"]

    try:
        install = subprocess.run(  # noqa: S603
            install_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        msg = (
            f"安装 wandb 失败 ({exc.__class__.__name__}: {exc})。\n"
            f"请手动执行:\n  {venv_python} -m pip install wandb\n"
            "或在前端 → 监控 → 关闭 W&B 后重启任务。"
        )
        on_event(TrainingEvent(
            type=EventType.error,
            payload={"source": "wandb-install", "error": msg},
        ))
        raise CompilationError(msg) from exc

    if install.returncode != 0:
        tail = (install.stderr or install.stdout or "").strip()
        msg = (
            f"安装 wandb 失败 (exit {install.returncode})。\n"
            f"请手动执行:\n  {venv_python} -m pip install wandb\n"
            f"或在前端 → 监控 → 关闭 W&B 后重启任务。\n"
            f"安装器输出 (尾部 500 字):\n{tail[-500:]}"
        )
        on_event(TrainingEvent(
            type=EventType.error,
            payload={"source": "wandb-install", "error": msg},
        ))
        raise CompilationError(msg)

    # Verify the install actually imports — uv reports success even when
    # the resolved wheel is broken for the target python ABI.
    verify_rc, verify_err = _probe()
    if verify_rc != 0:
        msg = (
            f"wandb 安装报告成功但仍无法 import:{verify_err[:300]}\n"
            f"venv: {venv_python}\n"
            "请手动检查 venv 完整性。"
        )
        on_event(TrainingEvent(
            type=EventType.error,
            payload={"source": "wandb-install", "error": msg},
        ))
        raise CompilationError(msg)

    on_event(
        TrainingEvent(
            type=EventType.log,
            payload={"source": "wandb-install", "message": "wandb 已装入 anima venv。"},
        )
    )
