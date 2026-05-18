"""Tests for the preview inference backend registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lorahub.core.inference import PromptSpec
from lorahub.core.inference.registry import (
    InferenceBackend,
    register_backend,
    registered_backend_names,
    resolve_backend,
    unregister_backend,
)


class _DummyBackend:
    name = "dummy"

    def __init__(self, *, arches: tuple[str, ...]) -> None:
        self._arches = arches
        self.rendered: list[Path] = []

    def is_available(self, *, arch: str) -> bool:
        return arch in self._arches

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.rendered.append(out_path)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    """Snapshot the registry, hide existing backends, restore on teardown.

    Prevents the real Anima/diffusers factories (registered at import
    time) from leaking into these unit tests.
    """
    import lorahub.core.inference.registry as reg

    snapshot = list(reg._REGISTRY)
    reg._REGISTRY.clear()
    try:
        yield
    finally:
        reg._REGISTRY.clear()
        reg._REGISTRY.extend(snapshot)


def test_register_and_resolve_first_match() -> None:
    backend_a = _DummyBackend(arches=("sdxl",))
    backend_b = _DummyBackend(arches=("flux",))
    register_backend("a", lambda **_: backend_a)
    register_backend("b", lambda **_: backend_b)

    resolved = resolve_backend(arch="sdxl")
    assert resolved is backend_a


def test_resolve_skips_unavailable_and_falls_through() -> None:
    backend_a = _DummyBackend(arches=("flux",))  # claims a different arch
    backend_b = _DummyBackend(arches=("sdxl",))
    register_backend("a", lambda **_: backend_a)
    register_backend("b", lambda **_: backend_b)

    resolved = resolve_backend(arch="sdxl")
    assert resolved is backend_b


def test_resolve_returns_none_when_no_backend_matches() -> None:
    register_backend(
        "image-only",
        lambda **_: _DummyBackend(arches=("sdxl",)),
    )
    assert resolve_backend(arch="hunyuan_video") is None


def test_resolve_skips_factory_returning_none() -> None:
    register_backend("opt-out", lambda **_: None)
    register_backend(
        "winner",
        lambda **_: _DummyBackend(arches=("sdxl",)),
    )
    resolved = resolve_backend(arch="sdxl")
    assert resolved is not None
    assert resolved.name == "dummy"


def test_resolve_swallows_factory_exceptions() -> None:
    def _broken(**_: Any) -> Any:
        raise RuntimeError("kaboom")

    register_backend("broken", _broken)
    register_backend(
        "winner",
        lambda **_: _DummyBackend(arches=("sdxl",)),
    )

    # The broken factory must not abort resolution.
    resolved = resolve_backend(arch="sdxl")
    assert resolved is not None


def test_register_backend_replaces_in_place() -> None:
    register_backend("a", lambda **_: _DummyBackend(arches=("sdxl",)))
    register_backend("b", lambda **_: _DummyBackend(arches=("sdxl",)))
    register_backend("a", lambda **_: _DummyBackend(arches=("flux",)))

    # Order preserved (a before b) but the factory was replaced.
    names = registered_backend_names()
    assert names == ["a", "b"]
    resolved = resolve_backend(arch="flux")
    assert resolved is not None
    resolved2 = resolve_backend(arch="sdxl")
    assert resolved2 is not None


def test_unregister_removes_backend() -> None:
    register_backend("a", lambda **_: _DummyBackend(arches=("sdxl",)))
    register_backend("b", lambda **_: _DummyBackend(arches=("sdxl",)))
    unregister_backend("a")

    assert registered_backend_names() == ["b"]


def test_dummy_backend_satisfies_protocol() -> None:
    """Any object exposing name + is_available + render passes the
    structural Protocol check used by the registry."""
    backend = _DummyBackend(arches=("sdxl",))
    assert isinstance(backend, InferenceBackend)


def test_stub_inference_still_works(tmp_path: Path) -> None:
    """StubInference is the fallback when no backend matches; ensure
    the registry refactor didn't break its render path."""
    from lorahub.core.inference import StubInference

    lora = tmp_path / "step100" / "lora.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"fake")
    out = tmp_path / "samples" / "step100_00.png"
    spec = PromptSpec(prompt="hello", index=0, width=64, height=64, seed=1)

    StubInference().render(
        lora_path=lora,
        spec=spec,
        out_path=out,
        default_steps=4,
        default_cfg=5.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
