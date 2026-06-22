from __future__ import annotations

import argparse
import os

from library.training import checkpoints


def test_epoch_and_step_checkpoints_live_under_output_name() -> None:
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
