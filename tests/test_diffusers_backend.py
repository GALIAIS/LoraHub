"""Tests for the generic diffusers preview backend.

Diffusers is not a hard dependency, so these tests stay strictly offline:
  * ``is_available`` is exercised both with diffusers absent (the common
    case in CI) and with diffusers monkey-patched in.
  * ``render`` is exercised with the diffusers import forced to fail so
    we can assert the lazy-import + InferenceUnavailable contract
    without ever touching ``from_pretrained`` (which would download
    multi-GB checkpoints).
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import pytest

from lorahub.core.inference import PromptSpec
from lorahub.core.inference.diffusers_backend import (
    DiffusersInferenceBackend,
    InferenceUnavailable,
    _factory,
)

# --------------------------------------------------------------------------- #
# is_available
# --------------------------------------------------------------------------- #


def test_is_available_false_for_video_arches() -> None:
    backend = DiffusersInferenceBackend(arch="hunyuan_video")
    assert backend.is_available(arch="hunyuan_video") is False


def test_is_available_false_for_anima() -> None:
    """Anima has its own dedicated backend; diffusers must opt out."""
    backend = DiffusersInferenceBackend(arch="anima")
    assert backend.is_available(arch="anima") is False


def test_is_available_false_when_diffusers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block the diffusers import and assert the gate fails closed."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("simulated missing diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Drop any cached import so the patched __import__ is hit.
    for mod in list(sys.modules):
        if mod == "diffusers" or mod.startswith("diffusers."):
            sys.modules.pop(mod, None)

    backend = DiffusersInferenceBackend(arch="sdxl")
    assert backend.is_available(arch="sdxl") is False


def test_is_available_true_when_diffusers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake diffusers module so the import gate passes."""
    fake = type(sys)("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    backend = DiffusersInferenceBackend(arch="sdxl")
    assert backend.is_available(arch="sdxl") is True


# --------------------------------------------------------------------------- #
# render — keep strictly offline
# --------------------------------------------------------------------------- #


def test_render_raises_unavailable_when_diffusers_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even if someone calls render past a stale is_available result,
    the lazy import inside render must still fail safely."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("simulated missing diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    for mod in list(sys.modules):
        if mod == "diffusers" or mod.startswith("diffusers."):
            sys.modules.pop(mod, None)

    backend = DiffusersInferenceBackend(arch="sdxl")
    lora = tmp_path / "step100" / "lora.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"fake")
    spec = PromptSpec(prompt="hi", index=0, width=64, height=64)
    out = tmp_path / "samples" / "x.png"
    with pytest.raises(InferenceUnavailable):
        backend.render(
            lora_path=lora,
            spec=spec,
            out_path=out,
            default_steps=4,
            default_cfg=5.0,
        )


# --------------------------------------------------------------------------- #
# Factory — recipe-aware base id resolution
# --------------------------------------------------------------------------- #


class _FakeBaseModel:
    def __init__(self, checkpoint: Any) -> None:
        self.checkpoint = checkpoint


class _FakeRecipe:
    def __init__(self, checkpoint: Any) -> None:
        self.base_model = _FakeBaseModel(checkpoint)


def test_factory_returns_none_for_unsupported_arch() -> None:
    assert _factory(arch="hunyuan_video", recipe=None, workspace=None) is None
    assert _factory(arch="anima", recipe=None, workspace=None) is None


def test_factory_returns_none_when_diffusers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("simulated missing diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    for mod in list(sys.modules):
        if mod == "diffusers" or mod.startswith("diffusers."):
            sys.modules.pop(mod, None)

    assert _factory(arch="sdxl", recipe=None, workspace=None) is None


def test_factory_picks_local_checkpoint_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A locally existing checkpoint path should override the default repo id."""
    fake_diffusers = type(sys)("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    ckpt = tmp_path / "model.safetensors"
    ckpt.write_text("fake")
    recipe = _FakeRecipe(checkpoint=ckpt)

    backend = _factory(arch="sdxl", recipe=recipe, workspace=tmp_path)
    assert backend is not None
    assert backend.base_model_id == str(ckpt)


def test_factory_treats_missing_local_path_as_repo_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-path-style checkpoint string should be passed through verbatim."""
    fake_diffusers = type(sys)("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    recipe = _FakeRecipe(checkpoint="some-org/some-repo")
    backend = _factory(arch="sdxl", recipe=recipe, workspace=tmp_path)
    assert backend is not None
    assert backend.base_model_id == "some-org/some-repo"
