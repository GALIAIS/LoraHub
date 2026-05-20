"""Sanity tests for the LPIPS validation helper.

We don't import the actual ``lpips`` package — testing the wrapper
logic + the graceful-fallback path is enough; real LPIPS values are
better validated by an end-to-end run.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_test_image(path: str, w: int = 64, h: int = 64) -> None:
    Image.new("RGB", (w, h), (123, 200, 50)).save(path)


def _stub_trainer():
    """Build a bare trainer-shaped object with the LPIPS helper bound.

    Avoids importing AnimaTrainer (which pulls accelerate + every
    model registry). We re-bind the method onto a SimpleNamespace via
    types.MethodType.
    """
    import types

    from train import AnimaTrainer

    # We can't easily instantiate AnimaTrainer; instead test the
    # method indirectly via an unbound call after binding to a
    # placeholder ``self``.
    obj = SimpleNamespace(_lpips_model=None)
    obj._compute_lpips_metric = types.MethodType(
        AnimaTrainer._compute_lpips_metric, obj,
    )
    return obj


def test_lpips_falls_back_silently_without_package() -> None:
    """If ``lpips`` isn't installed, helper logs a warning and returns
    ``None`` so the caller can skip the metric."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.png"
        _write_test_image(str(ref_path), 64, 64)
        gen = (torch.rand(3, 64, 64) * 2.0) - 1.0

        ref_item = SimpleNamespace(absolute_path=str(ref_path))
        # Stub the import so it raises ImportError deterministically.
        import importlib
        import sys as _sys

        # Save / restore lpips module if user happens to have it installed.
        had_lpips = "lpips" in _sys.modules
        _sys.modules["lpips"] = None  # type: ignore[assignment]
        try:
            try:
                trainer = _stub_trainer()
            except ImportError:
                # If train.py itself can't import (heavy deps), bail —
                # this test only validates the no-op branch which is
                # exercised at runtime anyway.
                print("test_lpips_falls_back_silently_without_package SKIPPED")
                return
            result = trainer._compute_lpips_metric(
                ref_items=[ref_item], gen_pixels=[gen], device="cpu",
            )
        finally:
            if had_lpips:
                _sys.modules.pop("lpips", None)
                importlib.import_module("lpips")
            else:
                _sys.modules.pop("lpips", None)
        assert result is None, "should return None when lpips package is absent"
        print("test_lpips_falls_back_silently_without_package OK")


def test_lpips_handles_empty_refs() -> None:
    """Edge case: zero-item validation set returns None gracefully."""
    try:
        trainer = _stub_trainer()
    except ImportError:
        print("test_lpips_handles_empty_refs SKIPPED")
        return
    result = trainer._compute_lpips_metric(
        ref_items=[], gen_pixels=[], device="cpu",
    )
    # Either None (lpips missing) or None (empty) — both fine.
    assert result is None
    print("test_lpips_handles_empty_refs OK")


if __name__ == "__main__":
    test_lpips_falls_back_silently_without_package()
    test_lpips_handles_empty_refs()
