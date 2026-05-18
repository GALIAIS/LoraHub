"""Integration test for the ``preview_unavailable`` event path.

When the recipe asks for a video arch (Wan / HunyuanVideo / ...) that no
registered backend can serve, ``_maybe_start_preview_worker`` must:
  * still spin up a worker (with ``StubInference``) so the rest of the
    event flow keeps producing ``sample_ready`` pings;
  * emit a ``preview_unavailable`` event so the UI can render a "your
    arch isn't supported" surface instead of leaving the user wondering
    why preview images look like coloured squares.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from lorahub.api.jobs_helpers import _maybe_start_preview_worker
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent


def _build_recipe(arch: str, prompts_file: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": "./model.safetensors", "arch": arch},
            "dataset": {"source": "./data"},
            "sampling": {
                "enabled": True,
                "enable_live_inference": True,
                "prompts_file": str(prompts_file),
                "inference_steps": 4,
                "inference_cfg": 5.0,
            },
        }
    )


def _make_prompts_file(tmp_path: Path) -> Path:
    f = tmp_path / "prompts.txt"
    f.write_text("preview prompt --w 64 --h 64 --d 1\n")
    return f


def test_video_arch_emits_preview_unavailable_and_uses_stub(tmp_path: Path) -> None:
    """``hunyuan_video`` is in the unsupported set; both registered
    backends (anima/diffusers) opt out, so the worker should still
    launch with ``StubInference`` *and* emit a ``preview_unavailable``
    event."""
    prompts = _make_prompts_file(tmp_path)
    cfg = _build_recipe("hunyuan_video", prompts)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "output").mkdir()

    events: list[TrainingEvent] = []
    stop_evt = threading.Event()
    handle = _maybe_start_preview_worker(
        cfg=cfg,
        workspace=workspace,
        job_id="J-VID",
        on_event=events.append,
        stop_evt=stop_evt,
    )
    try:
        assert handle is not None, "worker should have launched"
        thread, worker = handle

        # The unavailable event is emitted synchronously during setup,
        # so it must already be in the events list.
        unavailable = [
            e for e in events if e.type is EventType.preview_unavailable
        ]
        assert unavailable, "expected a preview_unavailable event"
        payload = unavailable[0].payload
        assert payload["arch"] == "hunyuan_video"
        assert "available_backends" in payload
        # The registry has both backends loaded by default — both must
        # appear in the audit list so the UI can show what was tried.
        assert "anima" in payload["available_backends"]
        assert "diffusers" in payload["available_backends"]
        assert "reason" in payload and payload["reason"]

        # The placeholder log event is emitted alongside.
        log_events = [
            e
            for e in events
            if e.type is EventType.log
            and e.payload.get("source") == "preview"
            and "no backend supports" in e.payload.get("message", "")
        ]
        assert log_events, "expected a placeholder log event"

        # Drop a checkpoint and let the worker render with the stub. We
        # don't assert on this strictly — just confirm the worker is
        # alive and processes data without crashing.
        ckpt_dir = workspace / "output" / "step100"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapter.safetensors").write_bytes(b"fake")
        worker.notify_checkpoint("step100")

        deadline = time.time() + 3
        while time.time() < deadline:
            if any(e.type is EventType.sample_ready for e in events):
                break
            time.sleep(0.05)
        sample_evs = [e for e in events if e.type is EventType.sample_ready]
        assert sample_evs, "stub fallback should still emit sample_ready"
    finally:
        stop_evt.set()
        if handle is not None:
            handle[0].join(timeout=2)


def test_supported_arch_does_not_emit_preview_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """When at least one backend claims the arch, the worker must NOT
    emit ``preview_unavailable`` — even if the chosen backend itself
    later fails (that's a separate failure mode)."""
    import lorahub.core.inference.registry as reg

    # Replace the registry with a single backend that always claims the
    # arch. Snapshot + restore so we don't pollute other tests.
    snapshot = list(reg._REGISTRY)
    reg._REGISTRY.clear()

    class _ClaimingBackend:
        name = "claiming"

        def is_available(self, *, arch: str) -> bool:
            return True

        def render(
            self, *, lora_path, spec, out_path, default_steps, default_cfg
        ) -> None:
            from PIL import Image  # noqa: PLC0415

            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (10, 20, 30)).save(out_path)

    reg.register_backend(
        "claiming", lambda **_: _ClaimingBackend()
    )

    try:
        prompts = _make_prompts_file(tmp_path)
        cfg = _build_recipe("sdxl", prompts)

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "output").mkdir()

        events: list[TrainingEvent] = []
        stop_evt = threading.Event()
        handle = _maybe_start_preview_worker(
            cfg=cfg,
            workspace=workspace,
            job_id="J-OK",
            on_event=events.append,
            stop_evt=stop_evt,
        )
        try:
            assert handle is not None
            unavailable = [
                e for e in events if e.type is EventType.preview_unavailable
            ]
            assert unavailable == [], "no unavailable event when backend claims arch"
        finally:
            stop_evt.set()
            if handle is not None:
                handle[0].join(timeout=2)
    finally:
        reg._REGISTRY.clear()
        reg._REGISTRY.extend(snapshot)
