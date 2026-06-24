from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def _load_checkpoints_module():
    path = Path(__file__).resolve().parents[1] / "library" / "training" / "checkpoints.py"
    spec = importlib.util.spec_from_file_location("_lorahub_checkpoints_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_epoch_and_step_checkpoints_live_under_output_name() -> None:
    checkpoints = _load_checkpoints_module()
    args = argparse.Namespace(output_name="demo")

    assert checkpoints.get_epoch_ckpt_name(args, ".safetensors", 2) == os.path.join(
        "demo",
        "demo-000002.safetensors",
    )
    assert checkpoints.get_step_ckpt_name(args, ".safetensors", 30) == os.path.join(
        "demo",
        "demo-step00000030.safetensors",
    )
    assert checkpoints.get_last_ckpt_name(args, ".safetensors") == "demo.safetensors"
    assert (
        checkpoints.get_checkpoint_ckpt_name(args, ".safetensors")
        == "demo-checkpoint.safetensors"
    )


def test_save_checkpoint_state_only_main_process_removes_old_dir(tmp_path) -> None:
    checkpoints = _load_checkpoints_module()
    args = argparse.Namespace(output_name="demo", output_dir=str(tmp_path))
    old_state = tmp_path / "demo-checkpoint-state"
    old_state.mkdir()
    (old_state / "owned-by-main-rank").write_text("x")
    calls: list[str] = []

    class _Accelerator:
        is_main_process = False

        def wait_for_everyone(self) -> None:
            calls.append("wait")

        def save_state(self, state_dir: str) -> None:
            calls.append(f"save:{os.path.basename(state_dir)}")

    checkpoints.save_checkpoint_state(args, _Accelerator())

    assert old_state.exists()
    assert calls == ["wait", "save:demo-checkpoint-state"]


def test_epoch_state_retention_removal_is_main_process_only(tmp_path) -> None:
    checkpoints = _load_checkpoints_module()
    args = argparse.Namespace(
        output_name="demo",
        output_dir=str(tmp_path),
        save_every_n_epochs=2,
        save_last_n_epochs_state=1,
        save_last_n_epochs=None,
    )
    old_state = tmp_path / "demo" / "demo-000002-state"
    old_state.mkdir(parents=True)
    calls: list[str] = []

    class _Accelerator:
        is_main_process = False

        def wait_for_everyone(self) -> None:
            calls.append("wait")

        def save_state(self, state_dir: str) -> None:
            calls.append(f"save:{os.path.basename(state_dir)}")

    checkpoints.save_and_remove_state_on_epoch_end(args, _Accelerator(), epoch_no=4)

    assert old_state.exists()
    assert calls == ["save:demo-000004-state", "wait"]


def test_step_state_retention_removal_is_main_process_only(tmp_path) -> None:
    checkpoints = _load_checkpoints_module()
    args = argparse.Namespace(
        output_name="demo",
        output_dir=str(tmp_path),
        save_every_n_steps=100,
        save_last_n_steps_state=100,
        save_last_n_steps=None,
    )
    old_state = tmp_path / "demo" / "demo-step00000100-state"
    old_state.mkdir(parents=True)
    calls: list[str] = []

    class _Accelerator:
        is_main_process = False

        def wait_for_everyone(self) -> None:
            calls.append("wait")

        def save_state(self, state_dir: str) -> None:
            calls.append(f"save:{os.path.basename(state_dir)}")

    checkpoints.save_and_remove_state_stepwise(args, _Accelerator(), step_no=201)

    assert old_state.exists()
    assert calls == ["save:demo-step00000201-state", "wait"]
