"""Tests for the attention-backend gating helpers.

The gating logic lives in :mod:`lorahub.api.system_stats` so the API
endpoint and any future CLI surface can share a single source of truth
about which kernels each compute-capability supports.
"""

from __future__ import annotations

import pytest

from lorahub.api.system_stats import (
    ALL_ATTENTION_BACKENDS,
    attention_backends_for_gpu,
)


def test_no_gpu_returns_safe_set() -> None:
    """No GPU detected -> only the PyTorch-native kernels."""
    out = attention_backends_for_gpu(None)
    assert "flash" not in out
    assert "flash3" not in out
    assert "xformers" not in out
    assert {"auto", "torch", "sdpa", "flex"}.issubset(out)


def test_empty_string_treated_as_unknown() -> None:
    """Empty / whitespace cap -> safe set (same as None)."""
    assert attention_backends_for_gpu("") == attention_backends_for_gpu(None)
    assert attention_backends_for_gpu("   ") == attention_backends_for_gpu(None)


def test_unparseable_cap_falls_back_to_safe() -> None:
    """Garbled compute-cap string -> conservative defaults."""
    assert attention_backends_for_gpu("bogus") == attention_backends_for_gpu(None)


def test_volta_gets_xformers_but_no_flash() -> None:
    """sm_70 (V100) — FA2 not supported, xformers fallback only."""
    out = attention_backends_for_gpu("7.0")
    assert "xformers" in out
    assert "flash" not in out
    assert "flash3" not in out


def test_ampere_includes_flash_but_not_flash3() -> None:
    """sm_80/86 - FA2 yes, FA3/FA4 no (Hopper-only)."""
    out = attention_backends_for_gpu("8.6")
    assert "flash" in out
    assert "flash3" not in out
    assert "flash4" not in out
    assert "xformers" in out


def test_ada_lovelace_matches_ampere_gating() -> None:
    """sm_89 = Ada Lovelace; same gating window as Ampere."""
    out = attention_backends_for_gpu("8.9")
    assert "flash" in out
    assert "flash3" not in out


def test_hopper_unlocks_flash3_and_flash4() -> None:
    """sm_90 - the only family that runs FA3."""
    out = attention_backends_for_gpu("9.0")
    assert "flash3" in out
    assert "flash4" in out
    assert "flash" in out


def test_blackwell_drops_flash3_and_xformers() -> None:
    """sm_100/120 - FA3 is Hopper-only, xformers wheels not yet shipped."""
    out = attention_backends_for_gpu("10.0")
    assert "flash3" not in out
    assert "xformers" not in out
    assert "flash4" in out
    assert "flash" in out

    out12 = attention_backends_for_gpu("12.0")
    assert "flash3" not in out12
    assert "flash4" in out12


@pytest.mark.parametrize("cap", ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0"])
def test_returned_options_are_subset_of_known_universe(cap: str) -> None:
    """Every option must be a known recipe-level value."""
    out = attention_backends_for_gpu(cap)
    assert set(out).issubset(ALL_ATTENTION_BACKENDS), (cap, out)
    # Every supported list always opens with the safe Pytorch-native set.
    assert out[:4] == ["auto", "torch", "sdpa", "flex"]


def test_all_attention_backends_matches_schema() -> None:
    """Catch drift between the schema enum and the gating universe."""
    from typing import get_args

    from lorahub.core.config.schema import AttentionConfig

    schema_choices = set(
        get_args(AttentionConfig.model_fields["training"].annotation)
    )
    assert schema_choices == set(ALL_ATTENTION_BACKENDS)
