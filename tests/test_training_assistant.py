"""Sanity tests for the training_assistant module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lorahub.api.training_assistant import (
    diagnose_failure,
    recommend_hyperparams,
)


# --- recommendations -----------------------------------------------------


def test_recommend_low_vram_path() -> None:
    s = recommend_hyperparams(
        dataset_size=120,
        gpu_vram_mb=8 * 1024,
        backend="anima_lora",
        target="character",
    )
    # Low VRAM → batch=1, ga ≥ 4
    assert s.batch_size == 1
    assert s.gradient_accumulation_steps >= 4
    # Optimizer should switch to 8bit
    assert s.optimizer_type == "AdamW8bit"
    # Anima extras
    assert s.extra_flags.get("ema") is True
    assert s.extra_flags.get("nan_guard") is True
    assert s.extra_flags.get("weighting_scheme") == "min_snr_rf"
    # Rationales fired
    assert any("VRAM" in r for r in s.rationale)


def test_recommend_high_vram_path() -> None:
    s = recommend_hyperparams(
        dataset_size=500,
        gpu_vram_mb=24 * 1024,
        backend="anima_lora",
        target="style",
    )
    assert s.batch_size == 4
    assert s.gradient_accumulation_steps == 1
    assert s.optimizer_type == "AdamW"
    # Style target → lower LR
    assert s.learning_rate == 5e-5
    # Mid-size dataset → rank 32
    assert s.network_dim == 32


def test_recommend_tiny_dataset_caps_capacity() -> None:
    s = recommend_hyperparams(
        dataset_size=20, gpu_vram_mb=16 * 1024, target="concept",
    )
    assert s.network_dim == 8


def test_recommend_serializes_clean() -> None:
    s = recommend_hyperparams(
        dataset_size=200, gpu_vram_mb=12 * 1024,
    )
    d = s.to_dict()
    # Must round-trip through JSON without weird types
    raw = json.dumps(d)
    loaded = json.loads(raw)
    assert loaded["batch_size"] == s.batch_size
    assert loaded["learning_rate"] == s.learning_rate


# --- diagnosis -----------------------------------------------------------


def test_diagnose_oom(tmp_path: Path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        "Epoch 3 starting...\n"
        "RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB\n",
        encoding="utf-8",
    )
    out = diagnose_failure(tmp_path, returncode=1, error=None)
    cats = [f["category"] for f in out["findings"]]
    assert "oom" in cats
    head = next(f for f in out["findings"] if f["category"] == "oom")
    assert "VRAM" in head["remediation"] or "batch" in head["remediation"].lower() or "checkpoint" in head["remediation"].lower()


def test_diagnose_nan_loss(tmp_path: Path) -> None:
    (tmp_path / "training.log").write_text(
        "step 100: loss=4.2\n"
        "step 101: loss=NaN — non-finite loss at global_step=101\n",
        encoding="utf-8",
    )
    out = diagnose_failure(tmp_path, returncode=1, error="non-finite loss")
    cats = [f["category"] for f in out["findings"]]
    assert "nan_loss" in cats


def test_diagnose_missing_module(tmp_path: Path) -> None:
    out = diagnose_failure(
        tmp_path,
        returncode=1,
        error="ModuleNotFoundError: No module named 'lpips'",
    )
    cats = [f["category"] for f in out["findings"]]
    assert "missing_module" in cats


def test_diagnose_user_cancel_is_info(tmp_path: Path) -> None:
    out = diagnose_failure(
        tmp_path,
        returncode=130,
        error="KeyboardInterrupt",
    )
    cats = [f["category"] for f in out["findings"]]
    assert "user_cancel" in cats
    head = next(f for f in out["findings"] if f["category"] == "user_cancel")
    assert head["severity"] == "info"


def test_diagnose_clean_run(tmp_path: Path) -> None:
    out = diagnose_failure(tmp_path, returncode=0, error=None)
    assert out["findings"] == []
    assert "no failure" in out["summary"].lower() or "exited" in out["summary"].lower() or "cleanly" in out["summary"].lower()


def test_diagnose_unknown_failure_path(tmp_path: Path) -> None:
    out = diagnose_failure(tmp_path, returncode=42, error="some opaque error")
    assert out["findings"]  # non-empty
    assert out["findings"][0]["category"] == "unknown"


def test_diagnose_reads_events_jsonl(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"event_type":"step","loss":1.0}\n'
        '{"event_type":"error","message":"CUDA out of memory: tried to allocate 2GB"}\n',
        encoding="utf-8",
    )
    out = diagnose_failure(tmp_path, returncode=1, error=None)
    cats = [f["category"] for f in out["findings"]]
    assert "oom" in cats


def test_diagnose_serializes_clean(tmp_path: Path) -> None:
    out = diagnose_failure(
        tmp_path,
        returncode=1,
        error="ModuleNotFoundError: No module named 'foo'",
    )
    raw = json.dumps(out)
    loaded = json.loads(raw)
    assert loaded["summary"]
    assert loaded["findings"][0]["category"] == "missing_module"
