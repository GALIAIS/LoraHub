"""Sanity tests for ``_compute_ema_telemetry``."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.training.ema import EMAModel  # noqa: E402
from library.training.loop import _compute_ema_telemetry  # noqa: E402


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(8, 16))
        self.b = nn.Parameter(torch.randn(4))

    def get_trainable_params(self):
        return [self.a, self.b]


def test_identical_shadow_returns_perfect_cos() -> None:
    net = _Net()
    ema = EMAModel(net)
    out = _compute_ema_telemetry(ema, net)
    assert "ema/cos_sim" in out and "ema/l2_dist" in out
    # Shadow == live at construction → cos=1, l2=0
    assert abs(out["ema/cos_sim"] - 1.0) < 1e-6
    assert out["ema/l2_dist"] < 1e-5
    print("test_identical_shadow_returns_perfect_cos OK")


def test_drifted_shadow_lowers_cos() -> None:
    net = _Net()
    ema = EMAModel(net, decay=0.5)
    # Mutate live params so shadow is now stale.
    with torch.no_grad():
        net.a.add_(2.0)
        net.b.mul_(-1.0)
    out = _compute_ema_telemetry(ema, net)
    # cos < 1, l2 > 0 — but no NaNs.
    assert out["ema/cos_sim"] < 1.0
    assert out["ema/l2_dist"] > 0.0
    assert out["ema/cos_sim"] == out["ema/cos_sim"]  # not NaN
    print("test_drifted_shadow_lowers_cos OK")


def test_param_count_mismatch_returns_empty() -> None:
    """If the network grew/shrunk vs the EMA snapshot, telemetry stays
    silent rather than crashing — matches ``EMAModel.step``'s policy."""
    net = _Net()
    ema = EMAModel(net)

    class _Bigger(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(8, 16))
            self.b = nn.Parameter(torch.randn(4))
            self.c = nn.Parameter(torch.randn(2))

        def get_trainable_params(self):
            return [self.a, self.b, self.c]

    bigger = _Bigger()
    out = _compute_ema_telemetry(ema, bigger)
    assert out == {}
    print("test_param_count_mismatch_returns_empty OK")


if __name__ == "__main__":
    test_identical_shadow_returns_perfect_cos()
    test_drifted_shadow_lowers_cos()
    test_param_count_mismatch_returns_empty()
