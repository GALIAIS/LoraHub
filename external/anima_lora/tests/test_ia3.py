"""IA3 — numerical correctness + save/load round-trip."""

from __future__ import annotations

import torch

from networks.lora_modules.ia3 import IA3Module


def _make_linear(in_dim: int, out_dim: int, *, seed: int = 0) -> torch.nn.Linear:
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        lin.weight.normal_(generator=g)
        lin.bias.normal_(generator=g)
    return lin


def test_ia3_init_is_identity():
    """At step 0 ``ℓ = 1`` so IA3 forward equals the unwrapped Linear."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=1)
    mod = IA3Module("ia3_init", org, lora_dim=4, alpha=4)
    expected = org(torch.randn(2, 8))  # snapshot before apply_to mutates org
    mod.apply_to()

    x = torch.randn(2, 8)
    # Re-run the original by calling org's saved forward directly via mod.
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_ia3_scales_per_output_channel():
    """Setting ``ℓ[i]`` rescales output channel i element-wise."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=2)
    mod = IA3Module("ia3_scale", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.ia3_weight.copy_(torch.tensor([2.5, 2.5, 1.0, 0.5, 0.5, 0.5]))
    mod.apply_to()

    x = torch.randn(3, 8)
    out = mod(x)
    expected = mod.org_forward(x) * mod.ia3_weight
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_ia3_get_weight_zero_multiplier_recovers_w0():
    """``get_weight(multiplier=0)`` interpolates ℓ → 1 → ΔW = 0."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=3)
    mod = IA3Module("ia3_m0", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.ia3_weight.copy_(torch.linspace(0.5, 1.5, 6))

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-7, rtol=0)


def test_ia3_fuse_unfuse_roundtrip_linear():
    """fuse + unfuse recovers W₀ and bias bit-equivalently for Linear."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=4)
    w0 = org.weight.detach().clone()
    b0 = org.bias.detach().clone()

    mod = IA3Module("ia3_fuse", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.ia3_weight.copy_(torch.linspace(0.4, 1.6, 6))

    mod.fuse_weight()
    expected_w = mod.ia3_weight.unsqueeze(1) * w0
    expected_b = mod.ia3_weight * b0
    torch.testing.assert_close(org.weight, expected_w, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(org.bias, expected_b, rtol=1e-5, atol=1e-6)

    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)
    torch.testing.assert_close(org.bias, b0, rtol=0, atol=0)


def test_ia3_merge_to_reproduces_fused_weight():
    """``merge_to`` writes the same tensor as ``fuse_weight`` produces."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=5)
    mod = IA3Module("ia3_merge", org, lora_dim=4, alpha=4)
    with torch.no_grad():
        mod.ia3_weight.copy_(torch.linspace(0.5, 1.5, 6))
    mod.fuse_weight()
    expected_w = org.weight.detach().clone()
    expected_b = org.bias.detach().clone()
    mod.unfuse_weight()

    fresh_org = _make_linear(8, 6, seed=5)
    fresh_mod = IA3Module("ia3_merge", fresh_org, lora_dim=4, alpha=4)
    sd = {"ia3_weight": mod.ia3_weight.detach().clone()}
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))

    torch.testing.assert_close(fresh_org.weight, expected_w, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fresh_org.bias, expected_b, rtol=1e-5, atol=1e-6)


def test_ia3_conv2d_shapes_and_identity():
    """Conv2d wrapper: identity at init, channel scaling at ℓ ≠ 1."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(6)
    org = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        org.weight.normal_(generator=g)
        org.bias.normal_(generator=g)
    mod = IA3Module("ia3_conv", org, lora_dim=2, alpha=2)
    mod.apply_to()

    x = torch.randn(1, 4, 8, 8)
    expected_id = mod.org_forward(x)
    got_id = mod(x)
    assert got_id.shape == expected_id.shape
    torch.testing.assert_close(got_id, expected_id, rtol=1e-5, atol=1e-6)

    with torch.no_grad():
        mod.ia3_weight.copy_(torch.linspace(0.5, 1.5, 6))
    out = mod(x)
    expected = mod.org_forward(x) * mod.ia3_weight.view(1, -1, 1, 1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_ia3_rejects_channel_scale():
    """IA3 doesn't act on the input axis — ``channel_scale`` raises."""
    org = _make_linear(8, 6, seed=7)
    import pytest

    with pytest.raises(ValueError, match="channel_scale"):
        IA3Module(
            "ia3_chan",
            org,
            lora_dim=4,
            alpha=4,
            channel_scale=torch.ones(8),
        )
