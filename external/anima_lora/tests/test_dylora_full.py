"""DyLoRA + Full — numerical correctness + save/load round-trip."""

from __future__ import annotations

import torch

from networks.lora_modules.dylora import DyLoRAModule
from networks.lora_modules.full import FullModule


# --- DyLoRA ---------------------------------------------------------------


def test_dylora_init_is_identity():
    """At step 0 lora_up is zero so ΔW = 0 regardless of truncation."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        org.weight.normal_()
        org.bias.normal_()
    mod = DyLoRAModule("dy_init", org, lora_dim=4, alpha=4)
    mod.apply_to()
    mod.eval()  # turn off rank truncation for determinism

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_dylora_eval_mode_uses_full_rank():
    """eval() ⇒ uses full ``lora_dim``; alpha-scale matches plain LoRA."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = DyLoRAModule("dy_full", org, lora_dim=4, alpha=4)
    mod.apply_to()
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
    mod.eval()

    x = torch.randn(3, 8)
    out = mod(x)

    delta = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
    expected = mod.org_forward(x) + (
        torch.nn.functional.linear(x, delta) * mod.multiplier * mod.scale
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


def test_dylora_train_mode_truncates_rank():
    """In train() each call samples a random b ∈ [1, lora_dim]; output stays bounded."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = DyLoRAModule("dy_train", org, lora_dim=4, alpha=4)
    mod.apply_to()
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
    mod.train()

    x = torch.randn(3, 8)
    # Many forward passes; outputs differ but stay finite.
    outs = [mod(x).detach() for _ in range(8)]
    for o in outs:
        assert torch.isfinite(o).all()
    # At least two passes should pick different b → outputs differ.
    assert any(
        not torch.allclose(outs[0], o, rtol=1e-3, atol=1e-3) for o in outs[1:]
    )


def test_dylora_get_weight_uses_full_rank():
    """``get_weight`` always returns the full-rank delta (used at fuse / save)."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = DyLoRAModule("dy_gw", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)

    delta = mod.get_weight()
    expected = mod.multiplier * (
        mod.lora_up.weight.float() @ mod.lora_down.weight.float()
    ) * mod.scale
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_dylora_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()
    mod = DyLoRAModule("dy_fuse", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)

    expected = w0 + mod._delta_full_rank()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


# --- Full -----------------------------------------------------------------


def test_full_init_is_identity():
    """Zero-init delta ⇒ forward equals unwrapped Linear at step 0."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        org.weight.normal_()
        org.bias.normal_()
    mod = FullModule("full_init", org, lora_dim=4, alpha=4)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_full_delta_is_free_parameter():
    """Setting ``delta`` directly to a known matrix produces additive shift."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = FullModule("full_delta", org, lora_dim=1, alpha=1)
    mod.apply_to()

    target = torch.randn(6, 8)
    with torch.no_grad():
        mod.delta.copy_(target)

    x = torch.randn(3, 8)
    out = mod(x)
    expected = mod.org_forward(x) + mod.multiplier * mod.scale * (target @ x.T).T
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_full_get_weight_zero_multiplier():
    """``get_weight(multiplier=0)`` returns zero — recovers W₀."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = FullModule("full_m0", org, lora_dim=1, alpha=1)
    with torch.no_grad():
        mod.delta.copy_(torch.randn_like(mod.delta))

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-7, rtol=0)


def test_full_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()
    mod = FullModule("full_fuse", org, lora_dim=1, alpha=1)
    with torch.no_grad():
        mod.delta.copy_(torch.randn_like(mod.delta) * 0.05)

    expected = w0 + mod._scaled_delta()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_full_merge_to_round_trip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()
    mod = FullModule("full_merge", org, lora_dim=1, alpha=1)
    with torch.no_grad():
        mod.delta.copy_(torch.randn_like(mod.delta) * 0.05)
    expected = w0 + mod._scaled_delta()

    fresh_org = torch.nn.Linear(8, 6, bias=False)
    fresh_org.weight.data.copy_(w0)
    fresh_mod = FullModule("full_merge", fresh_org, lora_dim=1, alpha=1)
    sd = {"delta": mod.delta.detach().clone()}
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


def test_dylora_full_reject_conv2d():
    import pytest

    org = torch.nn.Conv2d(4, 6, 3)
    with pytest.raises(ValueError, match="Conv2d"):
        DyLoRAModule("dy_conv", org, lora_dim=2, alpha=2)
    with pytest.raises(ValueError, match="Conv2d"):
        FullModule("full_conv", org, lora_dim=1, alpha=1)
