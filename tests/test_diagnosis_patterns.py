"""Fixture coverage for the diagnosis pattern catalogue.

Each rule in lorahub.api.diagnosis_patterns gets one fixture log line
that *should* match. The test asserts that:

1. ``diagnose_failure`` returns at least one finding for the synthetic
   log.
2. The finding's category equals the rule we expected to fire.

This catches accidental regressions when someone tweaks a regex (e.g.
fixing one false positive but losing the original target string) and
also documents the canonical example for each rule, which is useful
when adding new rules later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.api import diagnosis_patterns
from lorahub.api.training_assistant import diagnose_failure


# Canonical example log line for each rule. Each fixture is something
# we've actually seen in the wild (or a verbatim quote from upstream
# error messages).
_FIXTURES: dict[str, str] = {
    "oom": "RuntimeError: CUDA out of memory. Tried to allocate 1.20 GiB",
    "nan_loss": "WARNING: Loss is NaN at step 1234, skipping update",
    "missing_module": 'ModuleNotFoundError: No module named "diffusers"',
    "missing_safetensors": (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'C:/models/sdxl_base.safetensors'"
    ),
    "torch_compile_fail": (
        "torch._dynamo: BackendCompilerFailed: backend='inductor' raised: "
        "InductorError: ..."
    ),
    "data_loader_corrupt": (
        "PIL.UnidentifiedImageError: cannot identify image file 'foo.jpg'"
    ),
    "user_cancel": "Traceback (most recent call last): ... KeyboardInterrupt",
    "vram_pressure": "RuntimeError: CUDA error: out of memory",
    "ansi_encode": (
        "UnicodeEncodeError: 'ascii' codec can't encode characters in "
        "position 44-48: ordinal not in range(128)"
    ),
    "cjk_path_decode": (
        "UnicodeDecodeError: 'gbk' codec can't decode byte 0xa0 in position 7"
    ),
    "cuda_driver_mismatch": (
        "RuntimeError: CUDA driver version is insufficient for CUDA runtime "
        "version"
    ),
    "accelerate_config_missing": (
        "RuntimeError: accelerate config not found at "
        "/home/me/.cache/huggingface/accelerate/default_config.yaml"
    ),
    "permission_denied_write": (
        "PermissionError: [WinError 5] Access is denied: "
        "'E:\\\\runs\\\\sdxl-1\\\\state.pt'"
    ),
    "disk_full": "OSError: [Errno 28] No space left on device",
    "bitsandbytes_missing": (
        "RuntimeError: bitsandbytes is not installed; install via "
        "uv pip install bitsandbytes"
    ),
    "xformers_incompat": (
        "WARNING: xformers wasn't built with CUDA support, falling back"
    ),
    "safetensors_corrupt": (
        "safetensors_rust.SafetensorError: Error while deserializing header: "
        "InvalidHeaderDeserialization"
    ),
    "caption_missing": "ValueError: No caption file for image 0001.png",
    "vram_startup": (
        "RuntimeError: CUDA error: out of memory during cudnn init"
    ),
    "deepspeed_nccl": "RuntimeError: NCCL communicator was aborted on rank 0",
    "distributed_timeout": (
        "Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=42, OpType=ALLREDUCE)"
    ),
    "subprocess_returncode": (
        "subprocess.CalledProcessError: Command '['accelerate','launch']' "
        "returned non-zero exit status 1"
    ),
}


def _write_log(workspace: Path, text: str) -> None:
    """Drop a synthetic training.log so diagnose_failure can find it."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "training.log").write_text(text + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "category,sample",
    list(_FIXTURES.items()),
    ids=list(_FIXTURES.keys()),
)
def test_each_pattern_matches_its_canonical_example(
    tmp_path: Path, category: str, sample: str
) -> None:
    _write_log(tmp_path, sample)
    result = diagnose_failure(tmp_path, returncode=1)
    cats = {f["category"] for f in result["findings"]}
    assert category in cats, (
        f"pattern {category!r} did not match its fixture line; "
        f"fired={cats!r}"
    )


def test_every_pattern_has_a_fixture() -> None:
    """Sanity guard: adding a rule without a test should fail loudly."""
    rule_categories = {p[0] for p in diagnosis_patterns.get_patterns()}
    fixture_categories = set(_FIXTURES)
    missing = rule_categories - fixture_categories
    assert not missing, (
        f"the following diagnosis rules have no fixture entry: "
        f"{sorted(missing)!r}"
    )


def test_unknown_failure_falls_back_to_unknown_category(tmp_path: Path) -> None:
    """Lines that don't match any rule still surface a useful finding."""
    _write_log(tmp_path, "Some weird traceback we've never seen")
    result = diagnose_failure(tmp_path, returncode=1)
    cats = {f["category"] for f in result["findings"]}
    assert "unknown" in cats, result


def test_clean_exit_does_not_emit_findings(tmp_path: Path) -> None:
    _write_log(tmp_path, "Training finished cleanly. Saved model.safetensors.")
    result = diagnose_failure(tmp_path, returncode=0)
    assert result["findings"] == [], result


def test_called_process_error_argv_does_not_misfire_nan_loss(tmp_path: Path) -> None:
    """Regression: the anima_lora CalledProcessError repr listed both
    --nan_guard and --masked_loss on the same line. The old
    ``NaN.*loss`` regex greedily spanned them and labelled the run a
    "Loss became NaN" failure, drowning the real subprocess_returncode
    finding. The new pattern only fires on trainer narratives.
    """
    sample = (
        "subprocess.CalledProcessError: Command '['F:\\\\D\\\\LoraHub\\\\external\\\\"
        "anima_lora\\\\.venv\\\\Scripts\\\\python.exe', "
        "'F:\\\\D\\\\LoraHub\\\\external\\\\anima_lora\\\\train.py', "
        "'--method', 'lora', '--preset', 'low_vram', "
        "'--nan_guard', '--nan_guard_recover', '--nan_guard_max_consecutive', '5', "
        "'--cache_latents', '--cache_latents_to_disk', "
        "'--max_train_epochs', '8', '--masked_loss', "
        "'--save_every_n_epochs', '2', '--save_state']' "
        "returned non-zero exit status 1."
    )
    _write_log(tmp_path, sample)
    result = diagnose_failure(tmp_path, returncode=1)
    cats = {f["category"] for f in result["findings"]}
    assert "nan_loss" not in cats, (
        f"nan_loss should not fire on argv reprs containing --nan_guard/--masked_loss; "
        f"fired={cats!r}"
    )
    # The legitimate signal is still surfaced.
    assert "subprocess_returncode" in cats, cats
