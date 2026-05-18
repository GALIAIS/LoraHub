"""Smoke tests for DirectEdit V-injection (paper Eq. 13).

Doesn't exercise the real DiT — just verifies the patching mechanism, the
state-machine semantics, and the CFG-batch broadcasting in
``_VInjectionState``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from library.inference.directedit import (
    _resolve_t_inj_blocks,
    _v_injection_scope,
    _VInjectionState,
)


class _FakeAttention(nn.Module):
    """Surface-compatible stand-in for ``library.anima.models.Attention``."""

    def __init__(self) -> None:
        super().__init__()
        self.output_proj = nn.Identity()
        self.output_dropout = nn.Identity()

    def compute_qkv(self, x, context, rope_cos_sin=None):  # noqa: ARG002
        return x, x, x

    def forward(self, x, attn_params, context, rope_cos_sin=None):  # noqa: ARG002
        return x


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeAnima(nn.Module):
    def __init__(self, n: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(n)])


# ─── _resolve_t_inj_blocks ──────────────────────────────────────────────────


def test_resolve_default_skips_final_block():
    assert _resolve_t_inj_blocks(_FakeAnima(n=4), None) == {0, 1, 2}


def test_resolve_explicit_indices():
    assert _resolve_t_inj_blocks(_FakeAnima(n=4), [1, 3]) == {1, 3}


def test_resolve_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _resolve_t_inj_blocks(_FakeAnima(n=4), [0, 4])


# ─── _VInjectionState ───────────────────────────────────────────────────────


def test_set_rows_swaps_v_at_configured_block():
    """In-batch swap: ``v[tar_row] = v[src_row]`` on configured blocks."""
    state = _VInjectionState({0})
    v = torch.stack([
        torch.tensor([1.0, 2.0, 3.0]),    # row 0 (src)
        torch.tensor([10.0, 20.0, 30.0]),  # row 1 (tar)
    ])
    state.set_rows(src_row=0, tar_row=1)
    out = state.hook(0, v)

    # Row 1 now mirrors row 0; row 0 unchanged.
    assert torch.equal(out[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out[1], torch.tensor([1.0, 2.0, 3.0]))


def test_hook_does_not_mutate_input_in_place():
    """Hook clones before writing — input tensor must be left untouched."""
    state = _VInjectionState({0})
    v = torch.stack([torch.tensor([1.0]), torch.tensor([2.0])])
    v_orig = v.clone()
    state.set_rows(src_row=0, tar_row=1)
    state.hook(0, v)
    assert torch.equal(v, v_orig)


def test_block_outside_index_set_is_noop():
    state = _VInjectionState({0})
    state.set_rows(src_row=0, tar_row=1)
    v = torch.stack([torch.tensor([1.0]), torch.tensor([2.0])])
    out = state.hook(99, v)
    # Untouched: row 1 still differs from row 0.
    assert torch.equal(out, v)


def test_cfg_three_row_batch_swaps_only_tar():
    """CFG layout ``[neg_tar, cond_src, cond_tar]`` — only row 2 gets row 1's V."""
    state = _VInjectionState({0})
    v = torch.randn(3, 16, 8, 64)  # [B=3, L, H, D]
    v_orig = v.clone()
    state.set_rows(src_row=1, tar_row=2)
    out = state.hook(0, v)

    assert out.shape == v.shape
    assert torch.equal(out[0], v_orig[0])  # neg_tar untouched
    assert torch.equal(out[1], v_orig[1])  # cond_src untouched
    assert torch.equal(out[2], v_orig[1])  # cond_tar mirrors cond_src


def test_unset_rows_passes_through():
    """Default state (rows = None): no swap, no copy."""
    state = _VInjectionState({0})
    v = torch.stack([torch.tensor([1.0]), torch.tensor([2.0])])
    out = state.hook(0, v)
    assert torch.equal(out, v)


def test_only_one_row_set_passes_through():
    """Either row None disables the swap — guards against half-configured state."""
    state = _VInjectionState({0})
    v = torch.stack([torch.tensor([1.0]), torch.tensor([2.0])])

    state.set_rows(src_row=0, tar_row=None)
    assert torch.equal(state.hook(0, v), v)

    state.set_rows(src_row=None, tar_row=1)
    assert torch.equal(state.hook(0, v), v)


# ─── _v_injection_scope (patch + restore) ───────────────────────────────────


def test_scope_patches_only_selected_blocks():
    """Patching adds an instance-level `forward`; restore removes it."""
    m = _FakeAnima(n=4)

    # Pre-patch: every attn resolves `forward` via the class method.
    for i in range(4):
        assert "forward" not in m.blocks[i].self_attn.__dict__

    with _v_injection_scope(m, {0, 2}):
        assert "forward" in m.blocks[0].self_attn.__dict__
        assert "forward" not in m.blocks[1].self_attn.__dict__
        assert "forward" in m.blocks[2].self_attn.__dict__
        assert "forward" not in m.blocks[3].self_attn.__dict__

    # Instance attribute removed after scope — back to class-method resolution.
    for i in range(4):
        assert "forward" not in m.blocks[i].self_attn.__dict__


def test_scope_restores_on_exception():
    m = _FakeAnima(n=2)

    with pytest.raises(RuntimeError, match="boom"):
        with _v_injection_scope(m, {0, 1}):
            raise RuntimeError("boom")

    for i in range(2):
        assert "forward" not in m.blocks[i].self_attn.__dict__


def test_scope_clears_state_on_exit():
    """``set_rows`` mid-scope is reset to (None, None) on exit so a leaked
    state reference can't accidentally swap V on a subsequent unrelated forward."""
    m = _FakeAnima(n=2)
    with _v_injection_scope(m, {0}) as state:
        state.set_rows(src_row=0, tar_row=1)
        assert state.src_row == 0 and state.tar_row == 1

    assert state.src_row is None
    assert state.tar_row is None


def test_empty_block_set_is_inert():
    """t_inj=0 path: scope receives an empty set — nothing patched, nothing breaks."""
    m = _FakeAnima(n=2)
    with _v_injection_scope(m, set()):
        for i in range(2):
            assert "forward" not in m.blocks[i].self_attn.__dict__
