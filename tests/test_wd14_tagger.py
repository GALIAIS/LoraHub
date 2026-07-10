"""Tests for the WD14 tagger (network + ONNX mocked)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from lorahub.core.tagging import wd14


def test_tagging_package_exports_download_status() -> None:
    from lorahub.core.tagging import download_status
    from lorahub.core.tagging.download_status import snapshot

    assert download_status.snapshot is snapshot


def test_download_progress_honors_stop_request() -> None:
    from lorahub.core.tagging import download_status

    stop = False
    progress_type = download_status.tqdm_class_for(
        "owner/model",
        "model.onnx",
        lambda: stop,
    )
    progress = progress_type(total=10)
    stop = True

    with pytest.raises(InterruptedError, match="stopped by user"):
        progress.update(1)


@dataclass
class _FakeSpec:
    name: str = "input"
    shape: tuple = (1, 448, 448, 3)


class _FakeSession:
    def __init__(self, fixed_probs: np.ndarray) -> None:
        self._probs = fixed_probs
        self.calls: list[np.ndarray] = []

    def get_inputs(self) -> list[_FakeSpec]:
        return [_FakeSpec()]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(feeds["input"])
        return [self._probs[None, :]]


def _make_labels_csv(path: Path) -> None:
    path.write_text(
        "tag_id,name,category,count\n"
        "1,general,9,1\n"
        "2,sensitive,9,1\n"
        "10,1girl,0,1\n"
        "11,blue_hair,0,1\n"
        "12,smile,0,1\n"
        "20,akagi_(azur_lane),4,1\n",
        encoding="utf-8",
    )


def _make_image(path: Path, size: tuple[int, int] = (320, 240)) -> None:
    Image.new("RGB", size, (200, 100, 50)).save(path)


@pytest.fixture
def fake_tagger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> wd14.WD14Tagger:
    labels = tmp_path / "selected_tags.csv"
    _make_labels_csv(labels)

    paths = {"model.onnx": tmp_path / "model.onnx", "selected_tags.csv": labels}
    paths["model.onnx"].write_bytes(b"")

    def fake_download(repo_id: str, filename: str, **_: Any) -> str:
        return str(paths[filename])

    monkeypatch.setattr(wd14, "hf_download", fake_download)

    probs = np.array([0.1, 0.9, 0.95, 0.5, 0.30, 0.92], dtype=np.float32)
    session = _FakeSession(probs)

    class _FakeOrt:
        InferenceSession = lambda _self, _path, providers=None: session  # noqa: ARG005, E731

    import onnxruntime  # noqa: PLC0415
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **kw: session)

    return wd14.WD14Tagger()


def test_preprocess_pads_and_resizes(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _make_image(img, size=(100, 50))
    arr = wd14._preprocess_image(img, 64)
    assert arr.shape == (1, 64, 64, 3)
    assert arr.dtype == np.float32
    assert 0 <= arr.min() and arr.max() <= 255


def test_preprocess_does_not_normalize(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(img)
    arr = wd14._preprocess_image(img, 32)
    assert arr.max() > 200  # raw [0, 255], not [0, 1]


def test_tag_image_filters_by_threshold(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    img = tmp_path / "test.png"
    _make_image(img)

    result = fake_tagger.tag_image(img)

    general_names = {t.name for t in result.general}
    assert "1girl" in general_names
    assert "blue_hair" in general_names
    assert "smile" not in general_names
    assert "akagi_(azur_lane)" in {t.name for t in result.character}


def test_character_threshold_independent(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    fake_tagger.character_threshold = 0.99
    img = tmp_path / "test.png"
    _make_image(img)

    result = fake_tagger.tag_image(img)
    assert result.character == []
    assert result.general  # general unaffected


def test_rating_picks_argmax(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    img = tmp_path / "test.png"
    _make_image(img)

    result = fake_tagger.tag_image(img)
    assert result.rating is not None
    assert result.rating.name == "sensitive"


def test_caption_format(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    img = tmp_path / "test.png"
    _make_image(img)

    result = fake_tagger.tag_image(img)
    cap = result.caption()
    assert cap.startswith("akagi (azur lane), ")
    assert "blue hair" in cap

    cap_us = result.caption(underscores=True)
    assert "blue_hair" in cap_us
    assert "akagi_(azur_lane)" in cap_us


def test_tag_directory_writes_captions(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    for i in range(3):
        _make_image(tmp_path / f"img_{i}.png")

    fake_tagger.tag_directory(tmp_path, write_caption=True)

    for i in range(3):
        cap = (tmp_path / f"img_{i}.txt").read_text(encoding="utf-8")
        assert "1girl" in cap


def test_tag_directory_skips_existing(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    img = tmp_path / "img.png"
    _make_image(img)
    (tmp_path / "img.txt").write_text("existing", encoding="utf-8")

    results = fake_tagger.tag_directory(tmp_path, skip_existing=True)
    assert results == []
    assert (tmp_path / "img.txt").read_text() == "existing"


def test_tag_directory_honors_stop_before_each_image(
    tmp_path: Path, fake_tagger: wd14.WD14Tagger
) -> None:
    for i in range(3):
        img = tmp_path / f"img_{i}.png"
        _make_image(img)
        img.with_suffix(".txt").write_text("existing", encoding="utf-8")

    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(InterruptedError, match="stopped by user"):
        fake_tagger.tag_directory(tmp_path, skip_existing=True, should_stop=should_stop)


def test_tag_directory_overwrites_when_asked(
    tmp_path: Path, fake_tagger: wd14.WD14Tagger
) -> None:
    img = tmp_path / "img.png"
    _make_image(img)
    (tmp_path / "img.txt").write_text("existing", encoding="utf-8")

    results = fake_tagger.tag_directory(tmp_path, skip_existing=False)
    assert len(results) == 1
    assert (tmp_path / "img.txt").read_text() != "existing"


def test_progress_callback_invoked(tmp_path: Path, fake_tagger: wd14.WD14Tagger) -> None:
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.png")

    seen: list[Path] = []
    fake_tagger.tag_directory(tmp_path, on_progress=lambda p, _r: seen.append(p))
    assert {p.name for p in seen} == {"a.png", "b.png"}


# --- device / provider resolution -------------------------------------------------


def test_resolve_providers_auto_picks_cuda_when_available() -> None:
    p = wd14._resolve_providers(
        "auto", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert p == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_auto_falls_back_to_cpu() -> None:
    p = wd14._resolve_providers("auto", ["CPUExecutionProvider"])
    assert p == ["CPUExecutionProvider"]


def test_resolve_providers_cpu_forces_cpu_even_with_cuda() -> None:
    p = wd14._resolve_providers(
        "cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert p == ["CPUExecutionProvider"]


def test_resolve_providers_cuda_explicit_works() -> None:
    p = wd14._resolve_providers(
        "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert p == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_cuda_explicit_raises_when_missing() -> None:
    with pytest.raises(wd14.CudaUnavailableError, match="onnxruntime-gpu"):
        wd14._resolve_providers("cuda", ["CPUExecutionProvider"])


def test_resolve_providers_unknown_device_rejected() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        wd14._resolve_providers("tpu", ["CPUExecutionProvider"])
