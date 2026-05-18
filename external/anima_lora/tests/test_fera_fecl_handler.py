"""Tests for the in-handler FECL math (plan2 task #5).

The FeRA Frequency-Energy Consistency Loss lives in the loss-registry
handler ``library.training.losses._fera_fecl_loss``. plan2 task #6
retired ``networks/methods/fera.py`` and the legacy pre-computed
``ctx.aux['fecl_loss']`` path; only the in-handler entry point survives.
"""

from __future__ import annotations

import argparse
import types

import torch

from library.training.losses import (
    LossContext,
    _fera_fecl_loss,
    build_loss_composer,
)


def _make_inputs(B=2, C=4, H=64, W=64, seed=0):
    torch.manual_seed(seed)
    z_base = torch.randn(B, C, H, W)
    z_fera = torch.randn(B, C, H, W)
    z_target = torch.randn(B, C, H, W)
    return z_base, z_fera, z_target


def _make_ctx(model_pred, target, network, aux):
    """LossContext factory with the FECL-irrelevant fields stubbed out."""
    return LossContext(
        args=argparse.Namespace(),
        batch={},
        model_pred=model_pred,
        target=target,
        timesteps=torch.zeros(model_pred.shape[0]),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(model_pred.shape[0]),
        network=network,
        aux=aux,
    )


def test_fera_fecl_handler_returns_zero_when_disabled():
    """Weight=0 short-circuits to a zero scalar without computing anything."""
    z_base, z_fera, z_target = _make_inputs(seed=2)
    network = types.SimpleNamespace(fecl_weight=0.0)
    ctx = _make_ctx(z_fera, z_target, network, {"fera": {"z_base": z_base}})
    out = _fera_fecl_loss(ctx)
    assert out.shape == ()
    assert out.item() == 0.0


def test_fera_fecl_handler_returns_zero_when_aux_missing():
    """Weight>0 but no z_base in aux → 0 scalar (FECL is silently off)."""
    _, z_fera, z_target = _make_inputs(seed=3)
    network = types.SimpleNamespace(fecl_weight=0.5)
    ctx = _make_ctx(z_fera, z_target, network, {})
    out = _fera_fecl_loss(ctx)
    assert out.shape == ()
    assert out.item() == 0.0


def test_fera_fecl_handler_is_positive_on_random_inputs():
    """FECL is a sum of squared band-ratio differences — strictly positive
    whenever ``δ`` and ``r`` aren't proportional, which is the typical case
    on random inputs.
    """
    z_base, z_fera, z_target = _make_inputs(seed=1)
    cfg = types.SimpleNamespace(
        fera_fecl_weight=1.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    network = types.SimpleNamespace(fecl_weight=1.0, cfg=cfg)
    ctx = _make_ctx(z_fera, z_target, network, {"fera": {"z_base": z_base}})
    out = _fera_fecl_loss(ctx)
    assert out.shape == ()
    assert out.item() > 0.0


def test_fera_fecl_handler_applies_weight():
    """The handler must scale the raw scalar by ``fecl_weight`` — that's
    the single scaling-knob location.
    """
    z_base, z_fera, z_target = _make_inputs(seed=4)

    cfg2 = types.SimpleNamespace(
        fera_fecl_weight=2.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    cfg_ref = types.SimpleNamespace(
        fera_fecl_weight=1.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    net_ref = types.SimpleNamespace(fecl_weight=1.0, cfg=cfg_ref)
    net_w2 = types.SimpleNamespace(fecl_weight=2.0, cfg=cfg2)

    def _ctx(net):
        return _make_ctx(z_fera, z_target, net, {"fera": {"z_base": z_base}})

    ref = _fera_fecl_loss(_ctx(net_ref))
    w2 = _fera_fecl_loss(_ctx(net_w2))
    assert torch.allclose(w2, 2.0 * ref, atol=1e-6)


def test_build_loss_composer_activates_fera_fecl_on_stacked_experts():
    """The composer must activate ``fera_fecl`` when the network is a
    LoRANetwork with ``cfg.use_moe_style == 'independent_A'`` and
    ``fecl_weight > 0``.
    """
    cfg = types.SimpleNamespace(
        use_moe_style="independent_A", fera_fecl_weight=0.5
    )
    network = types.SimpleNamespace(
        cfg=cfg,
        fecl_weight=0.5,
        _ortho_reg_weight=0.0,
        _balance_loss_weight=0.0,
        contrastive_weight=0.0,
    )
    args = argparse.Namespace(
        method="lora",
        functional_loss_weight=0.0,
        multiscale_loss_weight=0.0,
        repa_weight=0.0,
        use_repa=False,
    )
    composer = build_loss_composer(args, network)
    assert "fera_fecl" in composer.active_losses


def test_build_loss_composer_skips_fera_fecl_on_plain_lora():
    """Without ``use_moe_style='independent_A'`` the composer must skip
    ``fera_fecl`` even when ``fecl_weight > 0`` is set on the network."""
    cfg = types.SimpleNamespace(use_moe_style=False, fera_fecl_weight=0.5)
    network = types.SimpleNamespace(
        cfg=cfg,
        fecl_weight=0.5,
        _ortho_reg_weight=0.0,
        _balance_loss_weight=0.0,
        contrastive_weight=0.0,
    )
    args = argparse.Namespace(
        method="lora",
        functional_loss_weight=0.0,
        multiscale_loss_weight=0.0,
        repa_weight=0.0,
        use_repa=False,
    )
    composer = build_loss_composer(args, network)
    assert "fera_fecl" not in composer.active_losses
