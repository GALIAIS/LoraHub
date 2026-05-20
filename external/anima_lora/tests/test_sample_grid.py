"""Sanity tests for the ``--sample_grid`` composite path."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.anima.training import _write_sample_grid  # noqa: E402


def _make_image(path: str, w: int, h: int, color=(128, 64, 32)) -> None:
    img = Image.new("RGB", (w, h), color)
    img.save(path)


def test_grid_concatenates_horizontally() -> None:
    with tempfile.TemporaryDirectory() as save_dir:
        # Pretend the trainer just wrote three samples for epoch 5.
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            _make_image(
                os.path.join(save_dir, f"run_e000005_{i:02d}_20260521120000_42.png"),
                100, 80, color,
            )
        args = argparse.Namespace(output_name="run", sample_grid=True)
        _write_sample_grid(args, save_dir, prompts=[1, 2, 3], epoch=5, steps=0)

        grid_path = os.path.join(save_dir, "grids", "run_e000005.png")
        assert os.path.isfile(grid_path), "grid wasn't written"
        with Image.open(grid_path) as grid:
            assert grid.size == (300, 80), f"unexpected grid size {grid.size}"
            rgb = grid.convert("RGB")
            assert rgb.getpixel((50, 40)) == (255, 0, 0)
            assert rgb.getpixel((150, 40)) == (0, 255, 0)
            assert rgb.getpixel((250, 40)) == (0, 0, 255)
        print("test_grid_concatenates_horizontally OK")


def test_grid_handles_missing_indices() -> None:
    """Skipping a prompt index (e.g. failed sample) leaves a gap that
    ``_write_sample_grid`` happily ignores — only the present indices
    are stitched."""
    with tempfile.TemporaryDirectory() as save_dir:
        _make_image(
            os.path.join(save_dir, "run_e000003_00_x_42.png"), 50, 50, (1, 0, 0)
        )
        # idx 1 missing
        _make_image(
            os.path.join(save_dir, "run_e000003_02_x_42.png"), 50, 50, (0, 0, 1)
        )
        args = argparse.Namespace(output_name="run", sample_grid=True)
        _write_sample_grid(args, save_dir, prompts=[1, 2, 3], epoch=3, steps=0)
        grid_path = os.path.join(save_dir, "grids", "run_e000003.png")
        assert os.path.isfile(grid_path)
        with Image.open(grid_path) as grid:
            assert grid.size == (100, 50)
        print("test_grid_handles_missing_indices OK")


def test_grid_step_path_no_epoch() -> None:
    """When ``epoch`` is None, we use the step-based num_suffix."""
    with tempfile.TemporaryDirectory() as save_dir:
        _make_image(
            os.path.join(save_dir, "run_001234_00_x.png"), 30, 30
        )
        args = argparse.Namespace(output_name="run", sample_grid=True)
        _write_sample_grid(args, save_dir, prompts=[1], epoch=None, steps=1234)
        grid_path = os.path.join(save_dir, "grids", "run_001234.png")
        assert os.path.isfile(grid_path)
        print("test_grid_step_path_no_epoch OK")


def test_grid_no_output_name() -> None:
    """``args.output_name = None`` is a valid configuration; the
    helper drops the leading underscore."""
    with tempfile.TemporaryDirectory() as save_dir:
        _make_image(
            os.path.join(save_dir, "e000001_00_x.png"), 30, 30
        )
        args = argparse.Namespace(output_name=None, sample_grid=True)
        _write_sample_grid(args, save_dir, prompts=[1], epoch=1, steps=0)
        grid_path = os.path.join(save_dir, "grids", "e000001.png")
        assert os.path.isfile(grid_path)
        print("test_grid_no_output_name OK")


if __name__ == "__main__":
    test_grid_concatenates_horizontally()
    test_grid_handles_missing_indices()
    test_grid_step_path_no_epoch()
    test_grid_no_output_name()
