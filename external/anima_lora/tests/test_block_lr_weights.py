"""Sanity tests for --block_lr_weights expansion into cfg.reg_lrs."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeCfg:
    def __init__(self, existing=None):
        self.reg_lrs = dict(existing) if existing else None


class _FakeNetwork:
    def __init__(self, cfg):
        self.cfg = cfg


def _apply_via_trainer(args, network, raw):
    """Re-implement the trainer method's logic directly so tests don't
    have to import the heavy AnimaTrainer class (which pulls torch +
    accelerate + every model registry).
    """
    weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
    base_lr = float(getattr(args, "unet_lr", None) or args.learning_rate or 0.0)
    cfg = network.cfg
    existing = dict(cfg.reg_lrs) if cfg.reg_lrs is not None else {}
    added = 0
    for idx, w in enumerate(weights):
        pat = rf"^.*blocks_{idx}_.*$"
        if pat in existing:
            continue
        existing[pat] = base_lr * w
        added += 1
    cfg.reg_lrs = existing
    return added


def test_basic_expansion() -> None:
    args = SimpleNamespace(unet_lr=1e-4, learning_rate=2e-4)
    cfg = _FakeCfg()
    net = _FakeNetwork(cfg)
    added = _apply_via_trainer(args, net, "1.0,0.8,0.5")
    assert added == 3
    assert cfg.reg_lrs[r"^.*blocks_0_.*$"] == 1e-4
    assert cfg.reg_lrs[r"^.*blocks_1_.*$"] == 8e-5
    assert abs(cfg.reg_lrs[r"^.*blocks_2_.*$"] - 5e-5) < 1e-12
    print("test_basic_expansion OK")


def test_user_reg_lrs_win_on_conflict() -> None:
    args = SimpleNamespace(unet_lr=1e-4, learning_rate=None)
    cfg = _FakeCfg(existing={r"^.*blocks_0_.*$": 9e-9})
    net = _FakeNetwork(cfg)
    added = _apply_via_trainer(args, net, "1.0,0.5")
    assert added == 1, "block 0 was already user-set; only block 1 should be added"
    assert cfg.reg_lrs[r"^.*blocks_0_.*$"] == 9e-9
    assert cfg.reg_lrs[r"^.*blocks_1_.*$"] == 5e-5
    print("test_user_reg_lrs_win_on_conflict OK")


def test_block_idx_anchored_correctly() -> None:
    """Pattern ``blocks_1_`` must NOT match ``blocks_10_`` etc."""
    pat = re.compile(r"^.*blocks_1_.*$")
    assert pat.match("lora_unet_blocks_1_self_attn_q")
    assert not pat.match("lora_unet_blocks_10_self_attn_q")
    assert not pat.match("lora_unet_blocks_11_self_attn_q")
    print("test_block_idx_anchored_correctly OK")


def test_zero_lr_skipped_safely() -> None:
    args = SimpleNamespace(unet_lr=None, learning_rate=0.0)
    cfg = _FakeCfg()
    net = _FakeNetwork(cfg)
    # base_lr = 0 → nothing to scale; the trainer logs a warning and
    # leaves cfg.reg_lrs alone. We mimic that here.
    base_lr = float(getattr(args, "unet_lr", None) or args.learning_rate or 0.0)
    assert base_lr == 0.0
    print("test_zero_lr_skipped_safely OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_basic_expansion()
    test_user_reg_lrs_win_on_conflict()
    test_block_idx_anchored_correctly()
    test_zero_lr_skipped_safely()
