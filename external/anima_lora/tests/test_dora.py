"""DoRA — numerical correctness + save/load round-trip."""

from __future__ import annotations

import torch

from networks.lora_modules.dora import DoRAModule
from networks.lora_modules.lora import rename_dora_keys


def _make_linear(in_dim: int, out_dim: int, *, seed: int = 0) -> torch.nn.Linear:
    """Deterministic Linear with non-trivial weight + bias for golden tests."""
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        lin.weight.normal_(generator=g)
        lin.bias.normal_(generator=g)
    return lin


def _column_norm(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 2:
        return weight.float().norm(dim=1, keepdim=True)
    return (
        weight.float()
        .reshape(weight.shape[0], -1)
        .norm(dim=1)
        .reshape(weight.shape[0], 1, 1, 1)
    )


def test_dora_init_is_identity():
    """At step 0 (zero-init lora_up + magnitude=‖W₀‖_c) DoRA must be a no-op.

    The LoRA delta is zero (lora_up.weight is zero-init); the effective
    weight collapses to ``‖W₀‖_c · W₀ / ‖W₀‖_c = W₀`` and the layer
    produces the same output as the unwrapped Linear.
    """
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=1)
    mod = DoRAModule(
        "test_dora_init",
        org,
        lora_dim=4,
        alpha=4,
        multiplier=1.0,
    )

    x = torch.randn(2, 8)
    expected = org(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_dora_magnitude_scales_output():
    """Doubling magnitude doubles the directional component.

    With lora_up still zero, the direction is ``W₀ / ‖W₀‖_c``. Setting
    ``magnitude = 2 · ‖W₀‖_c`` should give output = 2 · (W₀ - 0) ·  x +
    bias (i.e. the linear part is doubled, bias unchanged).
    """
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=2)
    mod = DoRAModule("test_dora_mag", org, lora_dim=4, alpha=4)

    with torch.no_grad():
        mod.magnitude.copy_(_column_norm(org.weight) * 2.0)

    x = torch.randn(3, 8)
    out = mod(x)
    expected = 2.0 * torch.nn.functional.linear(x, org.weight) + org.bias
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_dora_get_weight_returns_signed_delta():
    """``get_weight`` returns the effective DoRA delta vs W₀.

    With magnitude perturbed and lora_up zero, the delta should equal
    ``(m_new / ‖W₀‖_c - 1) · W₀`` exactly.
    """
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=3)
    mod = DoRAModule("test_dora_dw", org, lora_dim=4, alpha=4, multiplier=1.0)

    org_norm = _column_norm(org.weight)
    new_mag = org_norm + 0.1
    with torch.no_grad():
        mod.magnitude.copy_(new_mag)

    delta = mod.get_weight()
    expected = (new_mag / org_norm - 1.0) * org.weight.float()
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-6)


def test_dora_zero_multiplier_recovers_w0():
    """``get_weight(multiplier=0)`` must return zero (no delta vs W₀)."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=4)
    mod = DoRAModule("test_dora_m0", org, lora_dim=4, alpha=4)

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-7, rtol=0)


def test_dora_fuse_unfuse_roundtrip():
    """``fuse_weight()`` + ``unfuse_weight()`` recovers W₀ bit-equivalently."""
    torch.manual_seed(0)
    org = _make_linear(8, 6, seed=5)
    w0 = org.weight.detach().clone()
    mod = DoRAModule("test_dora_fuse", org, lora_dim=4, alpha=4)

    with torch.no_grad():
        # Make the LoRA leg + magnitude non-trivial so fuse actually changes
        # the weight.
        mod.lora_up.weight.normal_()
        mod.magnitude.add_(0.05)

    fused_dir = mod._effective_weight().to(org.weight.dtype)

    mod.fuse_weight()
    torch.testing.assert_close(org.weight, fused_dir, rtol=1e-5, atol=1e-6)

    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_dora_save_key_rename():
    """``rename_dora_keys`` maps ``.magnitude`` → ``.dora_scale``.

    LoraHub trains DoRA with the in-memory ``.magnitude`` Parameter
    name; ComfyUI's stock LoRA loader expects ``.dora_scale``. The
    rename is the standard save path's contract — verify it preserves
    the tensor identity.
    """
    sd = {
        "anima.layer.lora_down.weight": torch.randn(4, 8),
        "anima.layer.lora_up.weight": torch.zeros(6, 4),
        "anima.layer.alpha": torch.tensor(4.0),
        "anima.layer.magnitude": torch.randn(6, 1),
        "anima.layer._org_weight_norm": torch.randn(6, 1),
    }
    rename_dora_keys(sd)

    assert "anima.layer.dora_scale" in sd
    assert "anima.layer.magnitude" not in sd
    # Internal-only buffer shouldn't ship in checkpoints.
    assert "anima.layer._org_weight_norm" not in sd


def test_dora_merge_to_reproduces_effective_weight():
    """``merge_to`` lands the same tensor that ``forward`` evaluates.

    Construct a DoRA module, populate its trainable params, take the
    ``_effective_weight()``, then route the same params through
    ``merge_to`` and assert the org_module.weight matches.
    """
    torch.manual_seed(0)
    in_dim, out_dim, rank = 8, 6, 4
    org = _make_linear(in_dim, out_dim, seed=6)
    mod = DoRAModule("merge_test", org, lora_dim=rank, alpha=rank)

    with torch.no_grad():
        mod.lora_down.weight.normal_(std=0.1)
        mod.lora_up.weight.normal_(std=0.1)
        mod.magnitude.add_(0.02)

    expected = mod._effective_weight().to(org.weight.dtype)

    # Build a save-style state dict and merge.
    sd = {
        "lora_down.weight": mod.lora_down.weight.detach().clone(),
        "lora_up.weight": mod.lora_up.weight.detach().clone(),
        # Use the rename'd key to mirror the on-disk shape.
        "dora_scale": mod.magnitude.detach().clone(),
    }
    fresh_org = _make_linear(in_dim, out_dim, seed=6)
    fresh_mod = DoRAModule("merge_test", fresh_org, lora_dim=rank, alpha=rank)
    # IMPORTANT: merge_to uses self.org_module.weight as W₀, so the
    # fresh_mod must wrap the same Linear seed.
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))

    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-6)


def test_dora_conv2d_shapes():
    """Conv2d wrapper produces a sensible-shaped output and identity at init."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    org = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        org.weight.normal_(generator=g)
        org.bias.normal_(generator=g)
    mod = DoRAModule("conv_test", org, lora_dim=2, alpha=2)

    x = torch.randn(1, 4, 8, 8)
    expected = org(x)
    got = mod(x)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
