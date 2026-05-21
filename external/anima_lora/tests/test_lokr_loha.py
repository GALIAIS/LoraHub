"""LoKr / LoHA — numerical correctness + save/load round-trip."""

from __future__ import annotations

import torch

from networks.lora_modules.loha import LoHAModule
from networks.lora_modules.lokr import LoKrModule, _factorise


# --- LoKr -----------------------------------------------------------------


def test_lokr_factorise_picks_largest_divisor():
    # 24 = 1·24, 2·12, 3·8, 4·6 → factor=8 → a=8 (largest divisor ≤ 8).
    a, c = _factorise(24, factor=8)
    assert a == 8 and c == 3
    # factor=5 → a must be 4 (next divisor of 24 ≤ 5).
    a, c = _factorise(24, factor=5)
    assert a == 4 and c == 6
    # Prime out_dim → factor lands on 1 unless prime fits.
    a, c = _factorise(13, factor=8)
    assert a == 1 and c == 13


def test_lokr_init_is_identity():
    """At step 0 (lokr_w2_b zero) ΔW = 0, so forward ≡ unwrapped Linear."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        org.weight.normal_()
        org.bias.normal_()

    mod = LoKrModule("lokr_init", org, lora_dim=2, alpha=2, factor=4)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_lokr_delta_matches_kron_definition():
    """When all parameters are non-zero, ΔW equals (α/r) · kron(W₁, BA)."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = LoKrModule("lokr_delta", org, lora_dim=2, alpha=2, factor=2)

    with torch.no_grad():
        mod.lokr_w1.normal_()
        mod.lokr_w2_a.normal_()
        mod.lokr_w2_b.normal_()

    delta = mod.get_weight()
    expected = (
        mod.multiplier
        * mod.scale
        * torch.kron(
            mod.lokr_w1.float(),
            mod.lokr_w2_b.float() @ mod.lokr_w2_a.float(),
        )
    )
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)
    assert delta.shape == (6, 8)


def test_lokr_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        org.weight.normal_()
        org.bias.normal_()
    w0 = org.weight.detach().clone()

    mod = LoKrModule("lokr_fuse", org, lora_dim=2, alpha=2, factor=2)
    with torch.no_grad():
        mod.lokr_w1.normal_(std=0.05)
        mod.lokr_w2_a.normal_(std=0.05)
        mod.lokr_w2_b.normal_(std=0.05)

    expected = w0 + mod._delta()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_lokr_merge_to_round_trip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()

    mod = LoKrModule("lokr_merge", org, lora_dim=2, alpha=2, factor=2)
    with torch.no_grad():
        mod.lokr_w1.normal_(std=0.05)
        mod.lokr_w2_a.normal_(std=0.05)
        mod.lokr_w2_b.normal_(std=0.05)
    expected = w0 + mod._delta()

    fresh_org = torch.nn.Linear(8, 6, bias=False)
    fresh_org.weight.data.copy_(w0)
    fresh_mod = LoKrModule("lokr_merge", fresh_org, lora_dim=2, alpha=2, factor=2)
    sd = {
        "lokr_w1": mod.lokr_w1.detach().clone(),
        "lokr_w2_a": mod.lokr_w2_a.detach().clone(),
        "lokr_w2_b": mod.lokr_w2_b.detach().clone(),
    }
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


# --- LoHA -----------------------------------------------------------------


def test_loha_init_is_identity():
    """``hada_w2_b`` zero-init ⇒ Hadamard product = 0 at step 0."""
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=True)
    with torch.no_grad():
        org.weight.normal_()
        org.bias.normal_()
    mod = LoHAModule("loha_init", org, lora_dim=2, alpha=2)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_loha_delta_matches_hadamard_definition():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    mod = LoHAModule("loha_delta", org, lora_dim=2, alpha=2)

    with torch.no_grad():
        mod.hada_w1_a.normal_()
        mod.hada_w1_b.normal_()
        mod.hada_w2_a.normal_()
        mod.hada_w2_b.normal_()

    delta = mod.get_weight()
    w1 = mod.hada_w1_a.float() @ mod.hada_w1_b.float()
    w2 = mod.hada_w2_a.float() @ mod.hada_w2_b.float()
    expected = mod.multiplier * mod.scale * (w1 * w2)
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_loha_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()

    mod = LoHAModule("loha_fuse", org, lora_dim=2, alpha=2)
    with torch.no_grad():
        mod.hada_w1_a.normal_(std=0.05)
        mod.hada_w1_b.normal_(std=0.05)
        mod.hada_w2_a.normal_(std=0.05)
        mod.hada_w2_b.normal_(std=0.05)

    expected = w0 + mod._delta()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-6)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_loha_merge_to_round_trip():
    torch.manual_seed(0)
    org = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        org.weight.normal_()
    w0 = org.weight.detach().clone()

    mod = LoHAModule("loha_merge", org, lora_dim=2, alpha=2)
    with torch.no_grad():
        mod.hada_w1_a.normal_(std=0.05)
        mod.hada_w1_b.normal_(std=0.05)
        mod.hada_w2_a.normal_(std=0.05)
        mod.hada_w2_b.normal_(std=0.05)
    expected = w0 + mod._delta()

    fresh_org = torch.nn.Linear(8, 6, bias=False)
    fresh_org.weight.data.copy_(w0)
    fresh_mod = LoHAModule("loha_merge", fresh_org, lora_dim=2, alpha=2)
    sd = {
        "hada_w1_a": mod.hada_w1_a.detach().clone(),
        "hada_w1_b": mod.hada_w1_b.detach().clone(),
        "hada_w2_a": mod.hada_w2_a.detach().clone(),
        "hada_w2_b": mod.hada_w2_b.detach().clone(),
    }
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


def test_lokr_loha_reject_non_linear():
    """Both Phase-1 cuts are Linear-only — Conv2d should raise clearly."""
    import pytest

    org = torch.nn.Conv2d(4, 6, 3)
    with pytest.raises(ValueError, match="Conv2d"):
        LoKrModule("lokr_conv", org, lora_dim=2, alpha=2, factor=2)
    with pytest.raises(ValueError, match="Conv2d"):
        LoHAModule("loha_conv", org, lora_dim=2, alpha=2)
