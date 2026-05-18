"""Per-pool router diagnostics for the ChimeraHydra dual-pool routing.

Locks in the fix for the metric artifact where ``hydra/expert_usage`` —
implemented as argmax-histogram over the concat ``[π_c | π_f]`` — read out
as "some experts not used at all, some fixed, total != 2" on chimera runs.
The new ``chimera/content_usage/*`` + ``chimera/freq_usage/*`` keys use
mean-gates per pool, each summing to 1 (total 2), normalized entropy per
pool (each in [0, 1]), and skip the misleading argmax aggregation.
"""

from __future__ import annotations

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import FreqRouter, LoRANetwork
from networks.lora_modules import ChimeraHydraLoRAModule


def _make_minimal_chimera_network(
    K_c: int = 3,
    K_f: int = 3,
    fei_dim: int = 2,
    sigma_feature_dim: int = 0,
) -> LoRANetwork:
    """Hand-roll a tiny chimera LoRANetwork — same pattern as
    ``test_global_router._make_minimal_stacked_experts_network``, bypassing
    the DiT model-loading machinery."""
    cfg = LoRANetworkCfg(
        num_experts=K_c + K_f,
        use_chimera_hydra=True,
        num_experts_content=K_c,
        num_experts_freq=K_f,
        use_moe_style="shared_A",
        route_per_layer=True,
        router_source="fei",
        fei_feature_dim=fei_dim,
        sigma_feature_dim=sigma_feature_dim,
        router_hidden_dim=8,
        router_tau=1.0,
        freq_router_init_std=0.1,
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
    net._chimera_router_stats_cache = None
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
    net._use_chimera_hydra = True
    net._balance_loss_weight = 0.0
    net._balance_w_content = 1.0
    net._balance_w_freq = 1.0
    net.global_router = None

    for i in range(2):
        org = torch.nn.Linear(8, 8, bias=False)
        mod = ChimeraHydraLoRAModule(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts_content=K_c,
            num_experts_freq=K_f,
        )
        # BaseLoRAModule.apply_to() installs ``org_forward`` — the hand-rolled
        # network bypasses ``LoRANetwork.apply_to`` so call it here.
        mod.apply_to()
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)

    net._wire_shared_sigma_buffers()
    net._wire_shared_fei_buffers()
    net._wire_shared_routing_buffers()
    net._wire_shared_freq_routing_buffers()
    net.freq_router = FreqRouter(
        input_dim=fei_dim + sigma_feature_dim,
        num_freq_experts=K_f,
        hidden_dim=cfg.router_hidden_dim,
        tau=cfg.router_tau,
        init_std=cfg.freq_router_init_std,
    )
    net.add_module("freq_router", net.freq_router)
    return net


def _drive_forward(net: LoRANetwork, batch: int = 4) -> None:
    """Push a forward through every chimera module so ``_last_gate`` is set,
    and fire the freq router via ``set_freq_routing_weights`` so its
    ``_last_gates`` is populated."""
    net.train()
    # Drive FreqRouter directly to populate its ``_last_gates`` and broadcast
    # π_f to every chimera module's _freq_routing_weights buffer.
    fei = torch.randn(batch, net.cfg.fei_feature_dim)
    pi_f = net.freq_router(fei)
    net.set_freq_routing_weights(pi_f)
    # Now run a forward on each chimera module so its ``_last_gate`` is set.
    x = torch.randn(batch, 8)
    for lora in net.unet_loras:
        _ = lora(x)


def test_chimera_router_stats_per_pool_sums_to_one():
    """Each pool's mean-gate vector sums to ~1, so content+freq usage sums to ~2.

    Fixes the "total is not summed up to 2" observation: the legacy
    argmax-histogram emitted one mass per sample across the entire concat,
    making the dashboard read as if the network had a single 6-way softmax
    instead of two 3-way softmaxes.
    """
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    # Push the FreqRouter output layer off-zero so π_f isn't degenerate.
    with torch.no_grad():
        net.freq_router.net[-1].weight.normal_(std=0.5)
    _drive_forward(net, batch=4)
    stats = net.get_chimera_router_stats()

    content_usage = stats["content_usage"]
    freq_usage = stats["freq_usage"]
    assert len(content_usage) == 3
    assert len(freq_usage) == 3
    assert abs(sum(content_usage) - 1.0) < 1e-5, content_usage
    assert abs(sum(freq_usage) - 1.0) < 1e-5, freq_usage
    # Combined "total" matches user's mental model of sum-to-2.
    assert abs(sum(content_usage) + sum(freq_usage) - 2.0) < 1e-5


def test_chimera_router_stats_entropy_in_unit_range():
    """Per-pool entropy is normalized by ``log(K_pool)`` so a uniform pool
    reads ~1.0 (max) and a one-hot pool reads ~0.0 — unlike the legacy
    aggregate which divided the sum-to-2 vector's entropy by ``log(E)``
    and could exceed 1.0.
    """
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    # Force the FreqRouter to be near-uniform (small weights).
    with torch.no_grad():
        net.freq_router.net[-1].weight.zero_()
        net.freq_router.net[-1].bias.zero_()
    _drive_forward(net, batch=4)
    stats = net.get_chimera_router_stats()
    # Both pools are near-uniform at init, so both entropies hug 1.0.
    assert 0.0 <= stats["content_entropy"] <= 1.0 + 1e-5
    assert 0.0 <= stats["freq_entropy"] <= 1.0 + 1e-5
    assert stats["freq_entropy"] > 0.99  # exactly uniform


def test_chimera_router_stats_emits_in_metrics():
    """``metrics()`` swaps the ``hydra/expert_usage`` block for ``chimera/*``
    keys when ``_use_chimera_hydra=True`` — the misleading argmax-histogram
    keys are dropped to avoid confusing the dashboard.
    """
    from library.training.metrics import MetricContext

    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    with torch.no_grad():
        net.freq_router.net[-1].weight.normal_(std=0.5)
    _drive_forward(net, batch=4)
    out = net.metrics(MetricContext(args=None, network=net))

    # Per-pool keys present.
    for i in range(3):
        assert f"chimera/content_usage/{i}" in out
        assert f"chimera/freq_usage/{i}" in out
    assert "chimera/content_entropy" in out
    assert "chimera/freq_entropy" in out
    assert "chimera/content_margin" in out
    assert "chimera/freq_margin" in out

    # Legacy aggregate keys NOT emitted — those were the misleading metric.
    assert "hydra/expert_usage/0" not in out
    assert "hydra/router_entropy" not in out


def test_chimera_router_stats_cache_invalidated_by_clear():
    """``clear_step_caches`` resets the chimera stats cache so the next step
    recomputes from the freshly-cached gates (parallel to
    ``_router_stats_cache``)."""
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    with torch.no_grad():
        net.freq_router.net[-1].weight.normal_(std=0.5)
    _drive_forward(net, batch=4)
    s1 = net.get_chimera_router_stats()
    assert s1  # populated
    assert net._chimera_router_stats_cache is not None

    net.clear_step_caches()
    assert net._chimera_router_stats_cache is None
    # _last_gate cleared too, so a fresh fetch returns empty until the next
    # forward populates it again.
    s2 = net.get_chimera_router_stats()
    assert s2 == {}


def test_chimera_router_stats_empty_on_non_chimera_network():
    """``get_chimera_router_stats`` short-circuits to ``{}`` when the network
    isn't chimera — no spurious empty keys leak into a plain hydra run."""
    from tests.test_global_router import _make_minimal_stacked_experts_network

    net = _make_minimal_stacked_experts_network(num_experts=3, fei_dim=2)
    # Defensive: a non-chimera network may not even have the attr set.
    if not hasattr(net, "_chimera_router_stats_cache"):
        net._chimera_router_stats_cache = None
    net._use_chimera_hydra = False
    assert net.get_chimera_router_stats() == {}
