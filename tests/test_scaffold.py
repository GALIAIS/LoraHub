"""Tests for the config scaffolder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lorahub.core.config import scaffold


def _make_dataset(directory: Path, count: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"img_{i:03d}.png").write_bytes(b"")
    return directory


def test_pick_vram_tier_24gb_card() -> None:
    tier = scaffold.pick_vram_tier(24576)
    assert tier.rank == 64
    assert tier.batch_size >= 4


def test_pick_vram_tier_8gb_card() -> None:
    tier = scaffold.pick_vram_tier(8192)
    assert tier.rank == 16
    assert tier.batch_size == 1


def test_pick_vram_tier_low_falls_to_minimum() -> None:
    tier = scaffold.pick_vram_tier(2048)
    assert tier.rank == 4
    assert tier.grad_accum >= 4


def test_pick_num_repeats_inverse_to_image_count() -> None:
    assert scaffold.pick_num_repeats(5) == 10
    assert scaffold.pick_num_repeats(30) == 5
    assert scaffold.pick_num_repeats(100) == 2
    assert scaffold.pick_num_repeats(500) == 1


def test_pick_num_repeats_zero_defaults_to_10() -> None:
    assert scaffold.pick_num_repeats(0) == 10


def test_count_images_filters_extensions(tmp_path: Path) -> None:
    _make_dataset(tmp_path, 3)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    assert scaffold.count_images(tmp_path) == 3


def test_count_images_missing_dir(tmp_path: Path) -> None:
    assert scaffold.count_images(tmp_path / "nope") == 0


def test_detect_arch_from_filename() -> None:
    assert scaffold.detect_arch(Path("sdxl_base_1.0.safetensors")) == "sdxl"
    assert scaffold.detect_arch(Path("illustriousXL_v01.safetensors")) == "sdxl"
    assert scaffold.detect_arch(Path("ponyDiffusionXL.safetensors")) == "sdxl"
    assert scaffold.detect_arch(Path("noobaiXL_e2.safetensors")) == "sdxl"
    assert scaffold.detect_arch(Path("animagineXL_v3.safetensors")) == "sdxl"
    assert scaffold.detect_arch(Path("flux1-dev.safetensors")) == "flux"
    assert scaffold.detect_arch(Path("sd3-medium.safetensors")) == "sd3"
    assert scaffold.detect_arch(Path("v1-5-pruned.safetensors")) == "sd15"
    # Unknown checkpoints default to SDXL (current most common case).
    assert scaffold.detect_arch(Path("mystery.safetensors")) == "sdxl"


def test_detect_arch_variant_from_filename() -> None:
    assert scaffold.detect_arch_variant(Path("ponyDiffusionXL.safetensors")) == "pony"
    assert scaffold.detect_arch_variant(Path("illustriousXL_v01.safetensors")) == "illustrious"
    assert scaffold.detect_arch_variant(Path("noobaiXL_e2.safetensors")) == "noobai"
    assert scaffold.detect_arch_variant(Path("animagineXL_v3.safetensors")) == "animagine"
    # Vanilla SDXL has no variant token in its filename.
    assert scaffold.detect_arch_variant(Path("sdxl_base_1.0.safetensors")) == ""
    assert scaffold.detect_arch_variant(Path("mystery.safetensors")) == ""


def test_auto_scaffold_uses_explicit_vram(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "data", 30)
    cfg = scaffold.auto_scaffold(
        name="test", checkpoint=tmp_path / "sdxl.safetensors", dataset=ds, vram_mib=8192
    )
    assert cfg.base_model.arch == "sdxl"
    assert cfg.network.rank == 16
    assert cfg.schedule.batch_size == 1
    assert cfg.dataset.num_repeats == 5
    assert cfg.dataset.resolution == [1024, 1024]
    assert cfg.output.name == "test"


def test_auto_scaffold_falls_back_to_8gb_tier_when_no_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _make_dataset(tmp_path / "data", 100)
    monkeypatch.setattr(scaffold, "detect_gpu_vram_mib", lambda: None)
    cfg = scaffold.auto_scaffold(
        name="t", checkpoint=tmp_path / "sdxl.safetensors", dataset=ds
    )
    assert cfg.network.rank == 16
    assert cfg.dataset.num_repeats == 2


def test_auto_scaffold_sd15_uses_lower_resolution(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "data", 50)
    cfg = scaffold.auto_scaffold(
        name="t",
        checkpoint=tmp_path / "v1-5-pruned.safetensors",
        dataset=ds,
        vram_mib=8192,
    )
    assert cfg.base_model.arch == "sd15"
    assert cfg.dataset.resolution == [768, 768]


def test_auto_scaffold_pony_sets_variant_and_lr(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "data", 60)
    cfg = scaffold.auto_scaffold(
        name="t",
        checkpoint=tmp_path / "ponyDiffusionXL.safetensors",
        dataset=ds,
        vram_mib=8192,
    )
    # Backbone is still SDXL; variant is what tells callers it's Pony.
    assert cfg.base_model.arch == "sdxl"
    assert cfg.base_model.arch_variant == "pony"
    assert cfg.optimizer.lr.unet == 4.0e-4
    assert cfg.optimizer.lr.text_encoder == 2.0e-4
    # 8GB tier alpha is 8; variant raises it to 16.
    assert cfg.network.alpha == 16


def test_auto_scaffold_illustrious_sets_variant(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "data", 60)
    cfg = scaffold.auto_scaffold(
        name="t",
        checkpoint=tmp_path / "illustriousXL_v01.safetensors",
        dataset=ds,
        vram_mib=8192,
    )
    assert cfg.base_model.arch == "sdxl"
    assert cfg.base_model.arch_variant == "illustrious"
    assert cfg.optimizer.lr.unet == 4.0e-4


def test_auto_scaffold_noobai_and_animagine_share_pony_lr(tmp_path: Path) -> None:
    for fname, expected_variant in [
        ("noobaiXL_e2.safetensors", "noobai"),
        ("animagineXL_v3.safetensors", "animagine"),
    ]:
        ds = _make_dataset(tmp_path / f"data_{expected_variant}", 60)
        cfg = scaffold.auto_scaffold(
            name="t",
            checkpoint=tmp_path / fname,
            dataset=ds,
            vram_mib=8192,
        )
        assert cfg.base_model.arch == "sdxl"
        assert cfg.base_model.arch_variant == expected_variant
        assert cfg.optimizer.lr.unet == 4.0e-4
        assert cfg.optimizer.lr.text_encoder == 2.0e-4


def test_auto_scaffold_vanilla_sdxl_has_no_variant(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "data", 60)
    cfg = scaffold.auto_scaffold(
        name="t",
        checkpoint=tmp_path / "sdxl_base_1.0.safetensors",
        dataset=ds,
        vram_mib=8192,
    )
    assert cfg.base_model.arch == "sdxl"
    assert cfg.base_model.arch_variant == ""
    # No variant means we fall back to LRConfig defaults.
    assert cfg.optimizer.lr.unet == 1.0e-4


def test_detect_gpu_vram_returns_none_when_smi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scaffold.shutil, "which", lambda _name: None)
    assert scaffold.detect_gpu_vram_mib() is None


def test_detect_gpu_vram_parses_smi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaffold.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def fake_run(*_a: Any, **_k: Any):  # type: ignore[no-untyped-def]
        class R:
            returncode = 0
            stdout = "8188\n"
        return R()

    monkeypatch.setattr(scaffold.subprocess, "run", fake_run)
    assert scaffold.detect_gpu_vram_mib() == 8188


def test_detect_gpu_vram_returns_none_on_smi_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaffold.shutil, "which", lambda _name: "/fake/nvidia-smi")

    def fake_run(*_a: Any, **_k: Any):  # type: ignore[no-untyped-def]
        class R:
            returncode = 9
            stdout = ""
        return R()

    monkeypatch.setattr(scaffold.subprocess, "run", fake_run)
    assert scaffold.detect_gpu_vram_mib() is None
