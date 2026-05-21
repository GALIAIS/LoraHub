"""GLoRA + VeRA — numerical correctness."""

from __future__ import annotations

import torch

from networks.lora_modules.glora import GLoRAModule
from networks.lora_modules.vera import VeRAModule


def _make_linear(in_dim, out_dim, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        lin.weight.normal_(generator=g)
    return lin


# --- GLoRA --------------------------------------------------------------


def test_glora_init_is_identity():
    """``lora_up`` zero-init ⇒ ΔW = 0 regardless of gate."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=1)
    mod = GLoRAModule("g_init", org, lora_dim=4, alpha=4)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_glora_gate_one_matches_plain_lora():
    """With gate=1 GLoRA reduces to plain LoRA (W_up · W_down) · scale."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=2)
    mod = GLoRAModule("g_gate1", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
        # gate already 1.0 by init.

    delta = mod.get_weight()
    expected = mod.multiplier * (
        mod.lora_up.weight.float() @ mod.lora_down.weight.float()
    ) * mod.scale
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_glora_gate_zero_kills_rank():
    """Setting gate[i] = 0 zeros rank i's contribution."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=3)
    mod = GLoRAModule("g_gate0", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
        mod.glora_gate.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))

    delta = mod.get_weight()
    # Equivalent to half-rank LoRA on cols 0 and 2.
    up_cols = mod.lora_up.weight[:, [0, 2]].float()
    down_rows = mod.lora_down.weight[[0, 2], :].float()
    expected = mod.multiplier * (up_cols @ down_rows) * mod.scale
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_glora_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=4)
    w0 = org.weight.detach().clone()
    mod = GLoRAModule("g_fuse", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
        mod.glora_gate.copy_(torch.linspace(0.5, 1.5, 4))

    expected = w0 + mod._delta()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_glora_merge_to_round_trip():
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=5)
    w0 = org.weight.detach().clone()
    mod = GLoRAModule("g_merge", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
        mod.glora_gate.copy_(torch.linspace(0.5, 1.5, 4))
    expected = w0 + mod._delta()

    fresh_org = _make_linear(8, 6, seed=5)
    fresh_mod = GLoRAModule("g_merge", fresh_org, lora_dim=4, alpha=4)
    sd = {
        "lora_down.weight": mod.lora_down.weight.detach().clone(),
        "lora_up.weight": mod.lora_up.weight.detach().clone(),
        "glora_gate": mod.glora_gate.detach().clone(),
    }
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


# --- VeRA ---------------------------------------------------------------


def test_vera_init_is_identity():
    """λ_b init zero ⇒ ΔW = 0 regardless of A/B/λ_d."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=6)
    mod = VeRAModule("v_init", org, lora_dim=4, alpha=4)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_vera_delta_matches_definition():
    """ΔW = (α/r) · diag(λ_b) · B · diag(λ_d) · A."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=7)
    mod = VeRAModule("v_delta", org, lora_dim=4, alpha=4)
    # vera_A / vera_B already random-init in __init__.
    with torch.no_grad():
        mod.vera_lambda_b.copy_(torch.linspace(0.1, 1.0, 6))
        mod.vera_lambda_d.copy_(torch.linspace(0.5, 1.5, 4))

    delta = mod.get_weight()
    A = mod.vera_A.float()
    B = mod.vera_B.float()
    gated_a = A * mod.vera_lambda_d.float().unsqueeze(1)
    gated_b = B * mod.vera_lambda_b.float().unsqueeze(1)
    expected = mod.multiplier * (gated_b @ gated_a) * mod.scale
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_vera_random_matrices_are_buffers_not_params():
    """A and B must NOT receive gradients (they're frozen random)."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=8)
    mod = VeRAModule("v_buf", org, lora_dim=4, alpha=4)

    param_names = {n for n, _ in mod.named_parameters()}
    assert "vera_A" not in param_names
    assert "vera_B" not in param_names
    # Scale vectors ARE parameters.
    assert "vera_lambda_b" in param_names
    assert "vera_lambda_d" in param_names


def test_vera_zero_multiplier_recovers_w0():
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=9)
    mod = VeRAModule("v_m0", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.vera_lambda_b.fill_(0.5)
        mod.vera_lambda_d.fill_(0.5)

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=0)


def test_vera_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=10)
    w0 = org.weight.detach().clone()
    mod = VeRAModule("v_fuse", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.vera_lambda_b.fill_(0.3)
        mod.vera_lambda_d.fill_(0.5)

    expected = w0 + mod._delta()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_vera_merge_to_round_trip():
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=11)
    w0 = org.weight.detach().clone()
    mod = VeRAModule("v_merge", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.vera_lambda_b.fill_(0.3)
        mod.vera_lambda_d.fill_(0.5)
    expected = w0 + mod._delta()

    fresh_org = _make_linear(8, 6, seed=11)
    fresh_mod = VeRAModule("v_merge", fresh_org, lora_dim=4, alpha=4)
    sd = {
        "vera_A": mod.vera_A.detach().clone(),
        "vera_B": mod.vera_B.detach().clone(),
        "vera_lambda_b": mod.vera_lambda_b.detach().clone(),
        "vera_lambda_d": mod.vera_lambda_d.detach().clone(),
    }
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


def test_glora_vera_reject_conv2d():
    import pytest

    org = torch.nn.Conv2d(4, 6, 3)
    with pytest.raises(ValueError, match="Conv2d"):
        GLoRAModule("g_conv", org, lora_dim=2, alpha=2)
    with pytest.raises(ValueError, match="Conv2d"):
        VeRAModule("v_conv", org, lora_dim=2, alpha=2)
