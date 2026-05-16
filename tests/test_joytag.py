"""Tests for the JoyTag tagger.

The model is loaded via a small ``_load_vision_model`` indirection in
``joytag.py`` so we can monkey-patch in a dummy ``nn.Module`` and verify the
real preprocessing -> forward -> sigmoid -> threshold pipeline without
hitting the network or building a 100M-param ViT in CI.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from lorahub.core.tagging import joytag
from lorahub.core.tagging.base import BaseTagger


def _write_dummy_safetensors(path: Path) -> None:
    """Build a valid (parseable) safetensors header. No real tensors."""
    header = {
        "vision_model.embeddings.weight": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [0, 4],
        },
        "__metadata__": {"format": "pt"},
    }
    body = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(body)) + body + b"\x00\x00\x00\x00")


def _make_image(path: Path, size: tuple[int, int] = (320, 240)) -> None:
    Image.new("RGB", size, (123, 45, 200)).save(path)


@pytest.fixture
def joytag_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Mock ``hf_hub_download`` to point at a tmp_path-backed model + tags + config."""
    weights = tmp_path / "model.safetensors"
    tags = tmp_path / "top_tags.txt"
    config = tmp_path / "config.json"
    _write_dummy_safetensors(weights)
    tags.write_text("1girl\nblue_hair\nsmile\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "class": "ViT",
                "n_tags": 3,
                "image_size": joytag.INPUT_SIZE,
                "num_blocks": 2,
                "patch_size": 16,
                "d_model": 64,
                "mlp_dim": 128,
                "num_heads": 4,
                "stochdepth_rate": 0.0,
                "use_sine": True,
                "loss_type": "focal2",
            }
        ),
        encoding="utf-8",
    )

    paths = {"model.safetensors": weights, "top_tags.txt": tags, "config.json": config}

    def fake_download(repo_id: str, filename: str, **_: Any) -> str:
        assert repo_id == joytag.DEFAULT_MODEL
        return str(paths[filename])

    monkeypatch.setattr(joytag, "hf_hub_download", fake_download)
    return paths


def test_joytagger_satisfies_base_protocol() -> None:
    """Structural typing check — JoyTagger has the methods BaseTagger needs."""
    instance = joytag.JoyTagger()
    assert isinstance(instance, BaseTagger)


def test_load_without_torch_raises_clear_error(
    joytag_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without torch, ``load()`` must surface a ``JoyTagModelError`` that
    mentions torch — never a bare ``ImportError`` leaking out.
    """
    def boom(_device: str) -> Any:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(joytag, "_resolve_torch_device", boom)
    tagger = joytag.JoyTagger(device="cpu")
    with pytest.raises(joytag.JoyTagModelError) as exc:
        tagger.load()
    assert "torch" in str(exc.value).lower()


def test_load_and_tag_image_with_dummy_model(
    joytag_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end happy path — preprocessing -> forward -> sigmoid -> threshold.

    Mocks ``_load_vision_model`` with a tiny ``nn.Module`` whose logits are
    crafted so that, after sigmoid, the first and third tags clear the 0.4
    threshold and the middle one doesn't.
    """
    torch = pytest.importorskip("torch")
    from torch import nn

    # logits chosen so that sigmoid(2.0) ~= 0.88 and sigmoid(-2.0) ~= 0.12.
    fixed_logits = torch.tensor([[2.0, -2.0, 2.0]])

    class DummyViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._n_tags = 3

        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            return fixed_logits

    captured: dict[str, Any] = {}

    def fake_loader(*, config: dict, weights_path: Path, device: torch.device) -> nn.Module:
        captured["config"] = config
        captured["weights_path"] = weights_path
        captured["device"] = device
        m = DummyViT().to(device)
        m.eval()
        return m

    monkeypatch.setattr(joytag, "_load_vision_model", fake_loader)

    img = tmp_path / "x.png"
    _make_image(img, size=(80, 60))

    tagger = joytag.JoyTagger(device="cpu")
    tagger.load()

    # Sanity: load() consumed all three Hub files, populated provider, kept tag list.
    assert tagger._tag_names == ["1girl", "blue_hair", "smile"]
    assert tagger.active_provider == "cpu"
    assert captured["config"]["class"] == "ViT"
    assert captured["weights_path"] == joytag_paths["model.safetensors"]

    result = tagger.tag_image(img)
    names = [t.name for t in result.tags]
    # Highest-score tag first; "blue_hair" filtered out by threshold.
    assert names == ["1girl", "smile"]
    assert result.tags[0].score > 0.8
    assert result.tags[1].score > 0.8

    # predict_tags is the BaseTagger-facing flat-list adapter.
    assert tagger.predict_tags(img) == ["1girl", "smile"]


def test_caption_replaces_underscores_by_default(
    joytag_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caption formatting matches kohya conventions (spaces by default)."""
    torch = pytest.importorskip("torch")
    from torch import nn

    class DummyViT(nn.Module):
        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[5.0, 5.0, 5.0]])

    monkeypatch.setattr(
        joytag,
        "_load_vision_model",
        lambda **_: DummyViT().eval(),
    )
    img = tmp_path / "x.png"
    _make_image(img, size=(80, 60))

    tagger = joytag.JoyTagger(device="cpu")
    result = tagger.tag_image(img)
    assert result.caption() == "1girl, blue hair, smile"
    assert result.caption(underscores=True) == "1girl, blue_hair, smile"


def test_safetensors_param_summary_describes_header(joytag_paths: dict[str, Path]) -> None:
    summary = joytag._safetensors_param_summary(joytag_paths["model.safetensors"])
    assert "1 tensors" in summary
    assert "vision_model.embeddings.weight" in summary


def test_safetensors_param_summary_handles_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x00" * 16)
    summary = joytag._safetensors_param_summary(bad)
    # Either parses to "no tensors found" or surfaces a parse error message —
    # both are acceptable, neither should explode the caller.
    assert "tensors" in summary or "unreadable" in summary


def test_preprocess_pads_and_normalises(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    img = tmp_path / "x.png"
    _make_image(img, size=(100, 50))
    tensor = joytag._preprocess_image(img)
    assert tensor.shape == (3, joytag.INPUT_SIZE, joytag.INPUT_SIZE)
    arr = tensor.numpy()
    assert 0.0 <= arr.min() and arr.max() <= 1.0
    assert arr.dtype == np.float32


def test_iter_images_filters_by_extension(tmp_path: Path) -> None:
    _make_image(tmp_path / "keep.png")
    _make_image(tmp_path / "keep.jpg")
    (tmp_path / "skip.txt").write_text("hi", encoding="utf-8")
    found = {p.name for p in joytag._iter_images(tmp_path, recursive=False)}
    assert found == {"keep.png", "keep.jpg"}
