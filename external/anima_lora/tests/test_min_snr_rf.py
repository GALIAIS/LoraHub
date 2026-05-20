"""Sanity tests for Min-SNR-γ loss weighting on rectified flow."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.anima.training import (  # noqa: E402
    compute_loss_weighting_for_anima,
    set_min_snr_gamma,
)


def test_baselines_unchanged() -> None:
    sigmas = torch.linspace(0.1, 0.9, 5)
    set_min_snr_gamma(None)

    uniform = compute_loss_weighting_for_anima("uniform", sigmas)
    assert torch.allclose(uniform, torch.ones_like(sigmas))

    cos = compute_loss_weighting_for_anima("cosmap", sigmas)
    assert torch.all(cos > 0)
    print("test_baselines_unchanged OK")


def test_min_snr_rf_disabled_without_gamma() -> None:
    sigmas = torch.linspace(0.1, 0.9, 5)
    set_min_snr_gamma(None)
    w = compute_loss_weighting_for_anima("min_snr_rf", sigmas)
    assert torch.allclose(w, torch.ones_like(sigmas)), (
        "min_snr_rf without γ should fall back to uniform"
    )
    print("test_min_snr_rf_disabled_without_gamma OK")


def test_min_snr_rf_clamps_high_snr() -> None:
    """Low-σ (=high SNR) timesteps should be down-weighted toward γ/SNR."""
    set_min_snr_gamma(5.0)
    # σ=0.1 → SNR = (0.9/0.1)² = 81, much > γ=5 → weighting = 5/81 ≈ 0.062
    # σ=0.9 → SNR = (0.1/0.9)² ≈ 0.0123, much < γ=5 → weighting = 1.0 (clamp inactive)
    sigmas = torch.tensor([0.1, 0.5, 0.9])
    w = compute_loss_weighting_for_anima("min_snr_rf", sigmas)
    # σ=0.1 case
    expected_low = 5.0 / 81.0
    assert abs(w[0].item() - expected_low) < 1e-5, w[0].item()
    # σ=0.5 case: SNR=1, weighting = min(1, 5)/1 = 1
    assert abs(w[1].item() - 1.0) < 1e-5
    # σ=0.9 case: SNR≈0.0123, < γ → weighting = 1.0
    assert abs(w[2].item() - 1.0) < 1e-5
    print("test_min_snr_rf_clamps_high_snr OK")


def test_min_snr_rf_handles_endpoints() -> None:
    """σ near 0/1 must not divide by zero."""
    set_min_snr_gamma(5.0)
    sigmas = torch.tensor([1e-9, 0.5, 1.0 - 1e-9])
    w = compute_loss_weighting_for_anima("min_snr_rf", sigmas)
    assert torch.isfinite(w).all(), w
    # Cleanup
    set_min_snr_gamma(None)
    print("test_min_snr_rf_handles_endpoints OK")


if __name__ == "__main__":
    test_baselines_unchanged()
    test_min_snr_rf_disabled_without_gamma()
    test_min_snr_rf_clamps_high_snr()
    test_min_snr_rf_handles_endpoints()
