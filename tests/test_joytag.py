"""Tests for the JoyTag tagger.

The real model load is currently a deliberate ``JoyTagModelError`` (see the
TODO in ``lorahub/core/tagging/joytag.py``) — these tests cover everything
*around* that hole: the ``BaseTagger`` interface, image preprocessing, the
safetensors header inspection, and verifying ``load()`` raises the expected
error after the HF Hub download succeeds.
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
    """Mock ``hf_hub_download`` to point at a tmp_path-backed model + tags."""
    weights = tmp_path / "model.safetensors"
    tags = tmp_path / "top_tags.txt"
    _write_dummy_safetensors(weights)
    tags.write_text("1girl\nblue_hair\nsmile\n", encoding="utf-8")

    paths = {"model.safetensors": weights, "top_tags.txt": tags}

    def fake_download(repo_id: str, filename: str, **_: Any) -> str:
        assert repo_id == joytag.DEFAULT_MODEL
        return str(paths[filename])

    monkeypatch.setattr(joytag, "hf_hub_download", fake_download)
    return paths


def test_joytagger_satisfies_base_protocol() -> None:
    """Structural typing check — JoyTagger has the methods BaseTagger needs."""
    instance = joytag.JoyTagger()
    assert isinstance(instance, BaseTagger)


def test_load_raises_clear_error_with_safetensors_summary(
    joytag_paths: dict[str, Path],
) -> None:
    """`load()` must download both files, read the tag list, then surface a
    ``JoyTagModelError`` — either because torch isn't installed in the test
    env, or because the upstream Models.py architecture isn't ported yet.
    Both branches prove the dispatch contract."""
    tagger = joytag.JoyTagger(device="cpu")
    with pytest.raises(joytag.JoyTagModelError) as exc:
        tagger.load()
    msg = str(exc.value)
    weights_path = str(joytag_paths["model.safetensors"])
    # Either path is acceptable; the test environment dictates which one fires.
    assert weights_path in msg or "torch" in msg.lower()
    # Tag list is parsed before torch / model construction, so it should be
    # populated regardless of which branch tripped.
    assert tagger._tag_names == ["1girl", "blue_hair", "smile"]


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
