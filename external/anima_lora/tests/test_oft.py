"""Diag-OFT + BOFT — orthogonality + numerical correctness."""

from __future__ import annotations

import torch

from networks.lora_modules.boft import BOFTModule, _butterfly_permutation
from networks.lora_modules.diag_oft import DiagOFTModule, _cayley_orthogonal


def _make_linear(in_dim: int, out_dim: int, *, seed: int = 0) -> torch.nn.Linear:
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        lin.weight.normal_(generator=g)
    return lin


# --- Cayley primitive ---------------------------------------------------


def test_cayley_returns_orthogonal_matrix():
    """``(I-Q)(I+Q)^{-1}`` of a skew Q is orthogonal (R^T R = I)."""
    torch.manual_seed(0)
    A = torch.randn(3, 5, 5)
    Q = A - A.transpose(-1, -2)
    R = _cayley_orthogonal(Q)
    eye = torch.eye(5).expand_as(R)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(R @ R.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5)


def test_cayley_zero_skew_gives_identity():
    """Q = 0 ⇒ Cayley returns I."""
    Q = torch.zeros(2, 4, 4)
    R = _cayley_orthogonal(Q)
    eye = torch.eye(4).expand_as(R)
    torch.testing.assert_close(R, eye)


# --- Diag-OFT -----------------------------------------------------------


def test_diag_oft_init_is_identity():
    """Zero-init oft_skew ⇒ R = I, forward equals unwrapped Linear."""
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=1)  # square so R is on the host axis
    mod = DiagOFTModule("oft_init", org, lora_dim=4, alpha=1)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_diag_oft_block_diag_R_is_orthogonal():
    """Materialised R must be orthogonal (block-diag of Cayley R_k's)."""
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=2)
    mod = DiagOFTModule("oft_orth", org, lora_dim=4, alpha=1)
    with torch.no_grad():
        # Random skew → non-trivial R blocks.
        mod.oft_skew.normal_(std=0.1)

    R = mod._rotation_matrix()
    eye = torch.eye(8)
    torch.testing.assert_close(R.T @ R, eye, atol=1e-4, rtol=1e-4)
    # Block-diag: off-block entries are zero.
    R_blocks_zero = R.clone()
    for k in range(mod._num_blocks):
        i, j = k * mod._block_size, (k + 1) * mod._block_size
        R_blocks_zero[i:j, i:j] = 0
    torch.testing.assert_close(
        R_blocks_zero, torch.zeros_like(R), atol=1e-6, rtol=0
    )


def test_diag_oft_zero_multiplier_recovers_w0():
    """``get_weight(multiplier=0)`` returns zero — interpolation collapses."""
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=3)
    mod = DiagOFTModule("oft_m0", org, lora_dim=4, alpha=1)
    with torch.no_grad():
        mod.oft_skew.normal_(std=0.5)

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=0)


def test_diag_oft_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=4)
    w0 = org.weight.detach().clone()
    mod = DiagOFTModule("oft_fuse", org, lora_dim=4, alpha=1)
    with torch.no_grad():
        mod.oft_skew.normal_(std=0.1)

    expected = mod._effective_weight()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-5)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_diag_oft_merge_to_round_trip():
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=5)
    w0 = org.weight.detach().clone()
    mod = DiagOFTModule("oft_merge", org, lora_dim=4, alpha=1)
    with torch.no_grad():
        mod.oft_skew.normal_(std=0.1)

    expected = mod._effective_weight()
    fresh_org = _make_linear(8, 8, seed=5)
    fresh_mod = DiagOFTModule("oft_merge", fresh_org, lora_dim=4, alpha=1)
    sd = {"oft_skew": mod.oft_skew.detach().clone()}
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-5)


# --- BOFT --------------------------------------------------------------


def test_butterfly_permutation_is_a_permutation():
    """Each stage's permutation must permute (not duplicate / drop) indices."""
    for f in range(4):
        perm = _butterfly_permutation(16, f)
        assert perm.shape == (16,)
        # ``unique`` over a permutation has length ``out_dim``.
        assert perm.unique().numel() == 16
        # And each index is in range.
        assert perm.min() >= 0 and perm.max() < 16


def test_boft_init_is_identity():
    """Zero-init boft_skew ⇒ each B = I → R = I → forward = unwrapped."""
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=6)
    mod = BOFTModule("boft_init", org, lora_dim=2, alpha=1, boft_factors=3)
    mod.apply_to()

    x = torch.randn(2, 8)
    expected = mod.org_forward(x)
    got = mod(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_boft_R_is_orthogonal():
    """Composed R from butterfly stages remains orthogonal."""
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=7)
    mod = BOFTModule("boft_orth", org, lora_dim=2, alpha=1, boft_factors=3)
    with torch.no_grad():
        mod.boft_skew.normal_(std=0.1)

    R = mod._rotation_matrix()
    eye = torch.eye(8)
    torch.testing.assert_close(R.T @ R, eye, atol=1e-4, rtol=1e-4)


def test_boft_zero_multiplier_recovers_w0():
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=8)
    mod = BOFTModule("boft_m0", org, lora_dim=2, alpha=1, boft_factors=3)
    with torch.no_grad():
        mod.boft_skew.normal_(std=0.5)

    delta = mod.get_weight(multiplier=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=0)


def test_boft_fuse_unfuse_roundtrip():
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=9)
    w0 = org.weight.detach().clone()
    mod = BOFTModule("boft_fuse", org, lora_dim=2, alpha=1, boft_factors=3)
    with torch.no_grad():
        mod.boft_skew.normal_(std=0.1)

    expected = mod._effective_weight()
    mod.fuse_weight()
    torch.testing.assert_close(org.weight, expected.to(org.weight.dtype), rtol=1e-5, atol=1e-5)
    mod.unfuse_weight()
    torch.testing.assert_close(org.weight, w0, rtol=0, atol=0)


def test_boft_merge_to_round_trip():
    torch.manual_seed(0)
    org = _make_linear(8, 8, seed=10)
    mod = BOFTModule("boft_merge", org, lora_dim=2, alpha=1, boft_factors=3)
    with torch.no_grad():
        mod.boft_skew.normal_(std=0.1)
    expected = mod._effective_weight()

    fresh_org = _make_linear(8, 8, seed=10)
    fresh_mod = BOFTModule("boft_merge", fresh_org, lora_dim=2, alpha=1, boft_factors=3)
    sd = {"boft_skew": mod.boft_skew.detach().clone()}
    fresh_mod.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(fresh_org.weight, expected, rtol=1e-5, atol=1e-5)


def test_oft_reject_conv2d():
    import pytest

    org = torch.nn.Conv2d(4, 6, 3)
    with pytest.raises(ValueError, match="Conv2d"):
        DiagOFTModule("oft_conv", org, lora_dim=2, alpha=1)
    with pytest.raises(ValueError, match="Conv2d"):
        BOFTModule("boft_conv", org, lora_dim=2, alpha=1)
