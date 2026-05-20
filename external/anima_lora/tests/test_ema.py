"""Sanity: EMAModel decays toward live params and survives serialise round-trip."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.training.ema import EMAModel  # noqa: E402


class _FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.randn(8, 16))
        self.lora_up = nn.Parameter(torch.zeros(16, 8))
        # Frozen tail — should NOT be tracked.
        self.frozen = nn.Parameter(torch.randn(4), requires_grad=False)

    def get_trainable_params(self):
        return [self.lora_down, self.lora_up]


def test_ema_init_and_step() -> None:
    net = _FakeNetwork()
    ema = EMAModel(net, decay=0.9, use_num_updates=False, device="gpu")
    assert ema.n_params == 2, f"expected 2 trainable params, got {ema.n_params}"
    assert ema.param_names == ["lora_down", "lora_up"]
    # Initially shadow == live.
    for shadow, live in zip(ema.shadow_params, [net.lora_down, net.lora_up]):
        assert torch.equal(shadow, live.detach().to(torch.float32))

    # Mutate live and step.
    with torch.no_grad():
        net.lora_down.add_(torch.ones_like(net.lora_down))
        net.lora_up.add_(torch.ones_like(net.lora_up))

    ema.step(net)
    # decay=0.9, delta=1.0  -->  shadow_new = 0.9*shadow_old + 0.1*live_new
    expected_down = 0.9 * (net.lora_down - 1.0).to(torch.float32) + 0.1 * net.lora_down.to(torch.float32)
    assert torch.allclose(ema.shadow_params[0], expected_down, atol=1e-5)
    print("test_ema_init_and_step OK")


def test_ema_swap_restores() -> None:
    net = _FakeNetwork()
    ema = EMAModel(net, decay=0.99)
    snapshot = net.lora_down.detach().clone()

    # Step a few times to move shadow.
    for _ in range(3):
        with torch.no_grad():
            net.lora_down.add_(0.1)
        ema.step(net)
    assert not torch.allclose(snapshot, net.lora_down.detach())

    # Swap should temporarily put shadow into live, then restore.
    pre_swap = net.lora_down.detach().clone()
    with ema.swap(net):
        # Inside the swap we should see shadow values.
        assert torch.allclose(
            net.lora_down.detach().to(torch.float32), ema.shadow_params[0]
        )
    assert torch.allclose(net.lora_down.detach(), pre_swap)
    print("test_ema_swap_restores OK")


def test_ema_state_dict_roundtrip() -> None:
    net = _FakeNetwork()
    ema = EMAModel(net, decay=0.5)
    with torch.no_grad():
        net.lora_down.add_(2.0)
    ema.step(net)
    state = ema.state_dict()

    # Build a fresh EMA against an identical fresh network and load.
    net2 = _FakeNetwork()
    ema2 = EMAModel(net2, decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.num_updates == ema.num_updates
    for a, b in zip(ema.shadow_params, ema2.shadow_params):
        assert torch.allclose(a, b)
    print("test_ema_state_dict_roundtrip OK")


def test_ema_warmup_decay() -> None:
    net = _FakeNetwork()
    ema = EMAModel(net, decay=0.999, use_num_updates=True)
    # n=0  -> min(0.999, 1/10) = 0.1
    assert abs(ema._effective_decay() - 0.1) < 1e-6
    ema.num_updates = 100
    # min(0.999, 101/110) = 0.918... < 0.999, so warmup wins
    assert abs(ema._effective_decay() - 101 / 110) < 1e-6
    print("test_ema_warmup_decay OK")


if __name__ == "__main__":
    test_ema_init_and_step()
    test_ema_swap_restores()
    test_ema_state_dict_roundtrip()
    test_ema_warmup_decay()
