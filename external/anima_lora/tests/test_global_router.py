"""Smoke tests for the GlobalRouter wiring on plan2 task #4.

Covers:
  * GlobalRouter init is uniform-at-step-0 (zero output layer + softmax/τ).
  * ``set_routing_weights`` broadcasts to every routing-aware module.
  * ``set_fei`` end-to-end: fires the GlobalRouter and broadcasts.
  * Aliasing-recovery after a simulated ``Module._apply`` orphans the link
    (mirrors ``test_hydra_sigma_band.test_set_sigma_recovers_aliasing_after_to_device``).
  * Save-handler dispatch picks ``stacked_experts_global_fei`` when the
    cfg has ``use_moe_style="independent_A"``.
"""

from __future__ import annotations

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import GlobalRouter, LoRANetwork
from networks.lora_modules import (
    HydraLoRAModule,
    OrthoHydraLoRAModule,
    StackedExpertsLoRAModule,
)


def _make_minimal_stacked_experts_network(
    num_experts: int = 3,
    fei_dim: int = 2,
    ortho: bool = False,
) -> LoRANetwork:
    """Hand-roll a tiny LoRANetwork with two StackedExpertsLoRAModule modules.

    Bypasses the DiT model-loading machinery — same pattern as
    ``test_hydra_sigma_band._make_minimal_hydra_network``.
    """
    cfg = LoRANetworkCfg(
        num_experts=num_experts,
        use_moe_style="independent_A",
        route_per_layer=False,
        router_source="fei",
        fei_feature_dim=fei_dim,
        router_hidden_dim=32,
        router_tau=0.7,
        lora_dim=4,
        alpha=4.0,
    )
    net = LoRANetwork.__new__(LoRANetwork)
    torch.nn.Module.__init__(net)
    net.cfg = cfg
    net.unet_loras = []
    net.text_encoder_loras = []
    net.text_encoder_refts = []
    net.unet_refts = []
    net._last_sigma = None
    net._router_stats_cache = None
    net._sigma_router_hits = 0
    net._sigma_router_names = None
    net._sigma_router_re = None
    net._hydra_router_re = None
    net._hydra_router_names = None
    net._hydra_router_hits = 0
    net._hydra_router_misses = 0
    net._fei_router_hits = 0
    net._fei_router_re = None
    net._fei_router_names = None
    net.use_fei_router = True
    net.use_sigma_router = False
    net._channel_scale_misses = []
    net._channel_scale_hits = 0
    net._last_up_grad_stats = {}
    net._use_hydra = True
    net._balance_loss_weight = 0.0
    for i in range(2):
        org = torch.nn.Linear(8, 8, bias=False)
        mod = StackedExpertsLoRAModule(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            ortho=ortho,
        )
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)
    # Wire the aliased-buffer registry + the GlobalRouter (mirrors the tail
    # of ``LoRANetwork.__init__`` that we skipped by using ``__new__``).
    net._wire_shared_sigma_buffers()
    net._wire_shared_fei_buffers()
    net._wire_shared_routing_buffers()
    net.global_router = GlobalRouter(
        input_dim=fei_dim,
        num_experts=num_experts,
        hidden_dim=cfg.router_hidden_dim,
        tau=cfg.router_tau,
    )
    net.add_module("global_router", net.global_router)
    return net


def test_global_router_uniform_at_init():
    """Zero-init output layer + softmax/τ → exactly uniform over experts."""
    router = GlobalRouter(input_dim=4, num_experts=5, hidden_dim=8, tau=0.7)
    x = torch.randn(3, 4)
    gates = router(x)
    assert gates.shape == (3, 5)
    assert torch.allclose(gates, torch.full_like(gates, 1.0 / 5.0))
    # Side effects: _last_gates / _last_input populated and detached.
    assert router._last_gates is not None
    assert not router._last_gates.requires_grad
    assert router._last_input is not None


def test_set_routing_weights_broadcasts():
    """``set_routing_weights`` writes one gate tensor to every routing-aware module."""
    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    gates = torch.tensor([[0.5, 0.3, 0.2]])
    net.set_routing_weights(gates)
    # Every routing-aware module's _routing_weights now holds those gates,
    # and they all alias the same shared tensor.
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    for lora in net._routing_aware_loras:
        assert lora._buffers["_routing_weights"] is canonical
    assert torch.allclose(canonical, gates)


def test_clear_routing_weights_resets_to_uniform():
    net = _make_minimal_stacked_experts_network(num_experts=4, fei_dim=2)
    net.set_routing_weights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
    net.clear_routing_weights()
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(canonical, torch.full_like(canonical, 0.25))


def test_set_fei_fires_global_router_and_broadcasts():
    """End-to-end: caller pushes FEI via ``set_fei``, the GlobalRouter
    runs, and its gates land on every routing-aware module."""
    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    # Perturb router weights so its output is no longer the uniform init —
    # otherwise we can't distinguish "the router fired" from "the
    # placeholder was already uniform".
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=1.0)
        net.global_router.net[-1].bias.normal_(std=1.0)
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    # Should match what the router produces on this FEI.
    with torch.no_grad():
        expected = net.global_router(fei)
    assert torch.allclose(canonical, expected, atol=1e-6)


def test_set_routing_weights_recovers_aliasing_after_apply():
    """Regression: ``Module._apply`` (``.to(device)``) reallocates each
    routing-aware module's ``_routing_weights`` buffer independently,
    breaking the aliasing established by ``_wire_shared_routing_buffers``.

    Mirrors ``test_hydra_sigma_band.test_set_sigma_recovers_aliasing_after_to_device``
    for the new buffer. Without aliasing recovery, the in-place ``copy_``
    targets the orphaned shared tensor and per-module ``_routing_weights``
    stays at its uniform placeholder forever.
    """
    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    # Simulate Module._apply: rebind each module's buffer to an independent
    # clone, so the shared-attribute identity link breaks.
    for lora in net._routing_aware_loras:
        lora._buffers["_routing_weights"] = lora._buffers["_routing_weights"].clone()
    pre_canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert net._shared_routing_weights is not pre_canonical, (
        "test setup: aliasing must be broken before set_routing_weights runs"
    )
    gates = torch.tensor([[0.8, 0.1, 0.1]])
    net.set_routing_weights(gates)
    # All modules' live ``_routing_weights`` must hold the value the caller
    # passed, and aliasing must be re-established.
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert net._shared_routing_weights is canonical
    for lora in net._routing_aware_loras:
        assert lora._buffers["_routing_weights"] is canonical
        assert torch.allclose(lora._buffers["_routing_weights"], gates)


def test_stacked_experts_module_consumes_routing_weights():
    """StackedExpertsLoRAModule.forward weights the per-expert sum by
    ``_routing_weights``. Confirm the broadcast actually changes outputs.
    """
    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    mod = net.unet_loras[0]
    # Break zero-init expert ups so the adapter contribution is non-zero.
    with torch.no_grad():
        mod.lora_up_weight.normal_(std=0.1)
        mod.lora_down_weight.normal_(std=0.1)
    mod.apply_to()
    x = torch.randn(4, 8)
    # Uniform routing.
    net.clear_routing_weights()
    y_uniform = mod(x)
    # One-hot on expert 0.
    one_hot = torch.zeros(4, 3)
    one_hot[:, 0] = 1.0
    net.set_routing_weights(one_hot)
    y_e0 = mod(x)
    # One-hot on expert 1.
    one_hot[:] = 0.0
    one_hot[:, 1] = 1.0
    net.set_routing_weights(one_hot)
    y_e1 = mod(x)
    # Different gates must produce different outputs (with overwhelmingly
    # high probability for random expert weights).
    assert not torch.allclose(y_uniform, y_e0, atol=1e-4)
    assert not torch.allclose(y_e0, y_e1, atol=1e-4)


def test_stacked_experts_save_load_round_trip(tmp_path):
    """A hand-crafted state_dict in StackedExperts (free, independent-A)
    form should survive save → load → re-fuse to the original tensor
    shape. Covers ``_build_stacked_experts_state_dict`` (save) +
    ``_stack_lora_ups`` + ``_refuse_split_stacked_experts_keys`` (load).
    """
    from networks.lora_anima.loading import (
        _refuse_split_stacked_experts_keys,
        _stack_lora_ups,
    )
    from networks.lora_save import _build_stacked_experts_state_dict

    # Two adapters: one with a fused qkv attention prefix, one plain.
    # The fused one exercises the q/k/v split + re-fuse round trip.
    E = 3
    r = 4
    in_dim = 8
    out_per_letter = 8  # q, k, v each get out_per_letter rows
    out_qkv = out_per_letter * 3  # fused (E, sum_out, r)
    out_mlp = 12  # plain

    fused_name = "lora_unet_blocks_0_self_attn_qkv_proj"
    plain_name = "lora_unet_blocks_0_mlp_layer1"

    sd = {
        f"{fused_name}.lora_down_weight": torch.randn(E, r, in_dim),
        f"{fused_name}.lora_up_weight": torch.randn(E, out_qkv, r),
        f"{fused_name}.alpha": torch.tensor(float(r)),
        f"{plain_name}.lora_down_weight": torch.randn(E, r, in_dim),
        f"{plain_name}.lora_up_weight": torch.randn(E, out_mlp, r),
        f"{plain_name}.alpha": torch.tensor(float(r)),
    }
    # Snapshot pre-save (used for round-trip equality check on the plain leg).
    pre = {k: v.clone() for k, v in sd.items()}

    # Save: produces per-expert per-component lora_downs.{i} + lora_ups.{i}.
    saved = _build_stacked_experts_state_dict(sd, dtype=None)
    # The fused prefix is gone from the saved view; per-component prefixes are present.
    assert f"{fused_name}.lora_up_weight" not in saved
    base = "lora_unet_blocks_0_self_attn_"
    for letter in ("q", "k", "v"):
        for e in range(E):
            assert f"{base}{letter}_proj.lora_ups.{e}.weight" in saved
            assert f"{base}{letter}_proj.lora_downs.{e}.weight" in saved
    # Plain leg keeps its name but is split per-expert: lora_ups.{e} / lora_downs.{e}.
    for e in range(E):
        assert f"{plain_name}.lora_ups.{e}.weight" in saved
        assert f"{plain_name}.lora_downs.{e}.weight" in saved

    # Load: stack per-expert, then re-fuse q/k/v.
    loaded = _stack_lora_ups(dict(saved))
    loaded = _refuse_split_stacked_experts_keys(loaded)

    # Round-trip: fused tensors should match the originals bit-exactly.
    for key in (
        f"{fused_name}.lora_down_weight",
        f"{fused_name}.lora_up_weight",
        f"{plain_name}.lora_down_weight",
        f"{plain_name}.lora_up_weight",
    ):
        assert key in loaded, key
        assert torch.allclose(loaded[key], pre[key]), f"mismatch on {key}"


def test_metrics_emits_fera_keys_when_global_router_fired():
    """After ``set_fei`` fires the GlobalRouter, ``metrics`` emits the
    ``fera/router_entropy`` / ``fera/router_margin`` / ``fera/expert_usage/*``
    keys. Hand-rolling the network skips the model-loading machinery; the
    ``_use_hydra`` attr is set in the helper so the gate before the
    ``fera/*`` block is satisfied.
    """
    from library.training.metrics import MetricContext

    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=1.0)
    fei = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    net.set_fei(fei)
    out = net.metrics(MetricContext(args=None, network=net))
    assert "fera/router_entropy" in out
    assert "fera/router_margin" in out
    assert "fera/expert_usage/0" in out
    assert "fera/expert_usage/1" in out
    assert "fera/expert_usage/2" in out
    # Usage sums to ~1 (per-expert mean gate weight, softmax rows → 1).
    total = sum(out[f"fera/expert_usage/{i}"] for i in range(3))
    assert abs(total - 1.0) < 1e-5


def test_router_receives_gradient_from_expert_forward():
    """FeRA eq. 6-7: gates appear as live multipliers in the expert mixing,
    so plain L_denoise backprop must reach the GlobalRouter parameters via
    ``∂L/∂α``. Asserts that a downstream loss on the routed adapter output
    populates ``global_router.net[-1].weight.grad`` to a non-zero value.

    Regression for the pre-fix state where ``set_fei`` ran the router under
    ``torch.no_grad()`` and ``set_routing_weights`` detached the gates,
    leaving the router with zero gradient signal and stuck at zero-init
    (uniform 1/E) forever.
    """
    torch.manual_seed(0)
    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    # Break the zero-init symmetry on the router output layer so gates are
    # not exactly uniform (argmax-style metrics aside, gradient still flows
    # at uniform; this just makes the test less fragile to numeric ties).
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=0.1)
        net.global_router.net[-1].bias.normal_(std=0.1)
        # Make experts non-trivial so adapter output isn't constant.
        for lora in net._routing_aware_loras:
            lora.lora_up_weight.data.normal_(std=0.1)

    # Wire org_forward (apply_to) so the module's forward can run.
    for lora in net._routing_aware_loras:
        lora.apply_to()
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    x = torch.randn(1, 8, requires_grad=False)
    y = net._routing_aware_loras[0](x)
    loss = y.pow(2).sum()
    loss.backward()

    final_layer = net.global_router.net[-1]
    assert final_layer.weight.grad is not None, (
        "GlobalRouter final-layer.weight.grad is None — autograd path from "
        "expert forward to router is broken (set_fei or set_routing_weights "
        "is detaching the gates)."
    )
    assert final_layer.weight.grad.abs().sum().item() > 0.0, (
        "GlobalRouter final-layer.weight.grad is all zeros — gates are not "
        "carrying gradient through the broadcast."
    )


def _make_minimal_hydra_global_router_network(
    *,
    module_cls,
    num_experts: int = 3,
    fei_dim: int = 2,
    in_dim: int = 16,
    out_dim: int = 16,
    lora_dim: int = 4,
) -> LoRANetwork:
    """Hand-roll a tiny LoRANetwork with two ``shared_A`` Hydra-family modules
    built with ``use_global_router=True``.

    Mirrors ``_make_minimal_stacked_experts_network`` but for the
    ``use_moe_style="shared_A" + route_per_layer=False`` cell. The Hydra /
    OrthoHydra modules drop their per-layer router and consume gates broadcast
    from the network-level ``GlobalRouter``.
    """
    cfg = LoRANetworkCfg(
        num_experts=num_experts,
        use_moe_style="shared_A",
        route_per_layer=False,
        router_source="fei",
        fei_feature_dim=fei_dim,
        router_hidden_dim=32,
        router_tau=0.7,
        lora_dim=lora_dim,
        alpha=float(lora_dim),
    )
    net = LoRANetwork.__new__(LoRANetwork)
    torch.nn.Module.__init__(net)
    net.cfg = cfg
    net.unet_loras = []
    net.text_encoder_loras = []
    net.text_encoder_refts = []
    net.unet_refts = []
    net._last_sigma = None
    net._router_stats_cache = None
    net._sigma_router_hits = 0
    net._sigma_router_names = None
    net._sigma_router_re = None
    net._hydra_router_re = None
    net._hydra_router_names = None
    net._hydra_router_hits = 0
    net._hydra_router_misses = 0
    net._fei_router_hits = 0
    net._fei_router_re = None
    net._fei_router_names = None
    net.use_fei_router = True
    net.use_sigma_router = False
    net._channel_scale_misses = []
    net._channel_scale_hits = 0
    net._last_up_grad_stats = {}
    net._use_hydra = True
    net._balance_loss_weight = 0.0
    for i in range(2):
        org = torch.nn.Linear(in_dim, out_dim, bias=False)
        mod = module_cls(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            use_global_router=True,
        )
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)
    net._wire_shared_sigma_buffers()
    net._wire_shared_fei_buffers()
    net._wire_shared_routing_buffers()
    net.global_router = GlobalRouter(
        input_dim=fei_dim,
        num_experts=num_experts,
        hidden_dim=cfg.router_hidden_dim,
        tau=cfg.router_tau,
    )
    net.add_module("global_router", net.global_router)
    return net


def test_hydra_global_router_receives_gradient_from_expert_forward():
    """Same FeRA-style autograd contract for the shared_A path: ``set_fei``
    fires the GlobalRouter, gates land on every ``HydraLoRAModule(use_global_router=True)``
    via a live (non-detached) buffer reference, and ``L_denoise`` backprop
    populates ``global_router.net[-1].weight.grad``.

    Regression for the latent failure mode where ``HydraLoRAModule.set_routing_weights``
    (or the equivalent on OrthoHydra) detaches the broadcast tensor — the
    network-level set_routing_weights bypasses the module method today, but
    the module-level fallback path must not silently break the gradient path
    either.
    """
    torch.manual_seed(0)
    net = _make_minimal_hydra_global_router_network(
        module_cls=HydraLoRAModule, num_experts=3, fei_dim=2
    )
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=0.1)
        net.global_router.net[-1].bias.normal_(std=0.1)
        for lora in net._routing_aware_loras:
            # Break shared-A zero-init so the per-expert sum is non-constant
            # and the gate gradient has signal to land on.
            lora.lora_up_weight.data.normal_(std=0.1)

    for lora in net._routing_aware_loras:
        lora.apply_to()
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    x = torch.randn(1, 16, requires_grad=False)
    y = net._routing_aware_loras[0](x)
    loss = y.pow(2).sum()
    loss.backward()

    final_layer = net.global_router.net[-1]
    assert final_layer.weight.grad is not None, (
        "Hydra GlobalRouter final-layer.weight.grad is None — autograd path "
        "from expert forward to router is broken."
    )
    assert final_layer.weight.grad.abs().sum().item() > 0.0, (
        "Hydra GlobalRouter final-layer.weight.grad is all zeros — gates are "
        "not carrying gradient through the broadcast."
    )


def test_ortho_hydra_global_router_receives_gradient_from_expert_forward():
    """OrthoHydra equivalent. The current module-level ``set_routing_weights``
    contains a detach/clone fallback; this test exercises the live network
    path (``LoRANetwork.set_routing_weights`` via ``set_fei``) and asserts
    gradient flow survives end-to-end.
    """
    torch.manual_seed(0)
    # in/out ≥ E*r = 12 so OrthoHydra's disjoint-slice SVD path is taken.
    net = _make_minimal_hydra_global_router_network(
        module_cls=OrthoHydraLoRAModule,
        num_experts=3,
        fei_dim=2,
        in_dim=16,
        out_dim=16,
        lora_dim=4,
    )
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=0.1)
        net.global_router.net[-1].bias.normal_(std=0.1)
        for lora in net._routing_aware_loras:
            # Zero-init λ + zero-init S_p ⇒ ΔW=0; perturb λ so the routed
            # adapter output carries non-trivial signal.
            lora.lambda_layer.data.normal_(std=0.1)

    for lora in net._routing_aware_loras:
        lora.apply_to()
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    x = torch.randn(1, 16, requires_grad=False)
    y = net._routing_aware_loras[0](x)
    loss = y.pow(2).sum()
    loss.backward()

    final_layer = net.global_router.net[-1]
    assert final_layer.weight.grad is not None, (
        "OrthoHydra GlobalRouter final-layer.weight.grad is None — autograd "
        "path from expert forward to router is broken."
    )
    assert final_layer.weight.grad.abs().sum().item() > 0.0, (
        "OrthoHydra GlobalRouter final-layer.weight.grad is all zeros — gates "
        "are not carrying gradient through the broadcast."
    )


def test_clear_routing_weights_restores_uniform_hydra():
    """Module-level ``clear_routing_weights`` must restore the uniform 1/E
    placeholder for ``HydraLoRAModule(use_global_router=True)``.
    """
    net = _make_minimal_hydra_global_router_network(
        module_cls=HydraLoRAModule, num_experts=4, fei_dim=2
    )
    net.set_routing_weights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
    net.clear_routing_weights()
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(canonical, torch.full_like(canonical, 0.25))


def test_clear_routing_weights_restores_uniform_ortho_hydra():
    """Same contract for ``OrthoHydraLoRAModule(use_global_router=True)``."""
    net = _make_minimal_hydra_global_router_network(
        module_cls=OrthoHydraLoRAModule,
        num_experts=4,
        fei_dim=2,
        in_dim=16,
        out_dim=16,
        lora_dim=4,
    )
    net.set_routing_weights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
    net.clear_routing_weights()
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(canonical, torch.full_like(canonical, 0.25))


def test_clear_routing_weights_restores_uniform_stacked_experts():
    """StackedExperts equivalent — the existing
    ``test_clear_routing_weights_resets_to_uniform`` covers the network path;
    this pins the contract alongside the Hydra / OrthoHydra companions so
    the router_state.py refactor can't silently regress one of them.
    """
    net = _make_minimal_stacked_experts_network(num_experts=4, fei_dim=2)
    net.set_routing_weights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
    net.clear_routing_weights()
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(canonical, torch.full_like(canonical, 0.25))
