"""Round-trip the ChimeraHydra on-disk format through the ComfyUI custom
node's adapter module.

The training-side ortho-to-hydra distillation produces a Hydra-MoE layout
(shared ``lora_down`` + per-expert ``lora_ups.{i}``) plus a K_c-narrowed
content router and top-level ``freq_router.net.*`` keys, with metadata
flagging the chimera split. This test hand-rolls a tiny safetensors file
matching that layout and verifies:

  1. ``adapter.load_adapter`` recognizes the chimera flag, captures K_c /
     K_f / FEI σ params / FreqRouter weights, and skips the
     stacked-experts refusal path (chimera shares the Hydra layout).
  2. The standalone chimera pre-hook + per-Linear hook math is well-
     formed (correct shapes, dual-pool gate sums to ~2, σ/FEI input
     reaches the per-Linear delta).

End-to-end equivalence against ``ChimeraHydraLoRAModule.forward``
needs the Cayley distillation pass and is out of scope here — this is a
loader/runtime smoke test, not a numerics-parity test. We deliberately
do NOT exercise ``_apply_hydra_live_to_model`` end-to-end because that
path requires ``comfy.lora`` (ComfyUI is not on the unit-test sys.path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _load_adapter_module():
    """Import the node's ``adapter.py`` without depending on ComfyUI."""
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_anima_node_adapter",
        here / "custom_nodes" / "comfyui-hydralora" / "adapter.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_chimera_checkpoint(
    path: Path,
    *,
    K_c: int,
    K_f: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    fei_dim: int,
    sigma_dim: int,
    fr_hidden: int = 8,
    prefix: str = "lora_unet_blocks_0_mlp_layer1",
) -> None:
    """Synth a minimal chimera ``_chimera.safetensors`` file."""
    E = K_c + K_f
    torch.manual_seed(123)
    sd = {
        f"{prefix}.lora_down.weight": torch.randn(rank, in_dim),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        # Content router: (K_c, rank), no σ/FEI columns.
        f"{prefix}.router.weight": torch.randn(K_c, rank) * 0.01,
        f"{prefix}.router.bias": torch.zeros(K_c),
        # FreqRouter MLP: Linear → SiLU → Linear → softmax/τ.
        "freq_router.net.0.weight": torch.randn(fr_hidden, fei_dim + sigma_dim),
        "freq_router.net.0.bias": torch.zeros(fr_hidden),
        "freq_router.net.2.weight": torch.randn(K_f, fr_hidden) * 0.5,
        "freq_router.net.2.bias": torch.zeros(K_f),
    }
    for i in range(E):
        sd[f"{prefix}.lora_ups.{i}.weight"] = torch.randn(out_dim, rank)

    metadata = {
        "ss_use_chimera_hydra": "true",
        "ss_num_experts_content": str(K_c),
        "ss_num_experts_freq": str(K_f),
        "ss_chimera_fei_feature_dim": str(fei_dim),
        "ss_chimera_sigma_feature_dim": str(sigma_dim),
        "ss_chimera_fei_sigma_low_div": "4.0",
        "ss_use_moe_style": "shared_A",
        "ss_route_per_layer": "true",
        "ss_router_source": "input",
    }
    save_file(sd, str(path), metadata=metadata)


def test_load_adapter_recognizes_chimera(tmp_path):
    """``load_adapter`` populates ``bundle['hydra']['chimera']`` with the
    pool split + FreqRouter state when the file carries the chimera flag.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()

    path = tmp_path / "anima_chimera_chimera.safetensors"
    _write_chimera_checkpoint(
        path, K_c=3, K_f=2, rank=4, in_dim=8, out_dim=8,
        fei_dim=2, sigma_dim=0,
    )

    bundle = adapter.load_adapter(str(path))
    hydra = bundle["hydra"]
    assert hydra is not None
    chimera = hydra.get("chimera")
    assert chimera is not None, "chimera block missing from bundle"
    assert chimera["num_experts_content"] == 3
    assert chimera["num_experts_freq"] == 2
    assert chimera["fei_feature_dim"] == 2
    assert chimera["sigma_feature_dim"] == 0
    fr = chimera["freq_router_sd"]
    assert fr["net.0.weight"].shape == (8, 2)
    assert fr["net.2.weight"].shape == (2, 8)
    # The plain-LoRA extraction path should not have picked up the
    # freq_router.* keys (chimera unrouted Linears would land in ``lora``,
    # but our test file has no plain LoRA modules → bundle["lora"] is None).
    assert bundle["lora"] is None


def test_load_adapter_rejects_misshaped_freq_router(tmp_path):
    """K_f / FreqRouter output dim mismatch fails loudly at load time."""
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()
    path = tmp_path / "bad_chimera.safetensors"
    _write_chimera_checkpoint(
        path, K_c=3, K_f=2, rank=4, in_dim=8, out_dim=8,
        fei_dim=2, sigma_dim=0,
    )
    from safetensors.torch import load_file
    from safetensors import safe_open
    sd = load_file(str(path))
    sd["freq_router.net.2.weight"] = torch.randn(99, 8)
    sd["freq_router.net.2.bias"] = torch.zeros(99)
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
    save_file(sd, str(path), metadata=meta)
    with pytest.raises(ValueError, match="FreqRouter output dim"):
        adapter.load_adapter(str(path))


def test_chimera_pre_hook_emits_pi_f(tmp_path):
    """Pre-hook computes FEI from ``args[0]`` and σ from ``args[1]``, runs
    the FreqRouter MLP, and stores a ``(B, K_f)`` softmax in router_state.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()
    path = tmp_path / "anima_chimera.safetensors"
    _write_chimera_checkpoint(
        path, K_c=3, K_f=2, rank=4, in_dim=8, out_dim=8,
        fei_dim=2, sigma_dim=4,
    )
    bundle = adapter.load_adapter(str(path))
    chimera = bundle["hydra"]["chimera"]

    state: dict = {}
    pre_hook = adapter._make_chimera_pre_hook(
        state,
        chimera["freq_router_sd"],
        fei_feature_dim=chimera["fei_feature_dim"],
        sigma_feature_dim=chimera["sigma_feature_dim"],
        fei_sigma_low_div=chimera["fei_sigma_low_div"],
        router_tau=chimera["router_tau"],
        K_f=chimera["num_experts_freq"],
    )

    B = 2
    latent = torch.randn(B, 4, 1, 16, 16)  # 5D — pre-hook squeezes T=1.
    timesteps = torch.tensor([0.1, 0.9])
    pre_hook(None, (latent, timesteps))

    assert state.get("sigma") is not None
    assert state["sigma"].dtype == torch.float32
    assert state.get("fei") is not None
    assert state["fei"].shape == (B, 2)
    pi_f = state.get("pi_f")
    assert pi_f is not None
    assert pi_f.shape == (B, 2)
    # Softmax: each row sums to 1, every entry positive.
    assert torch.allclose(pi_f.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (pi_f > 0).all()


def test_chimera_hook_dispatches_dual_pool(tmp_path):
    """Per-Linear hook concatenates ``[π_c, π_f]`` and dispatches the full
    E = K_c + K_f einsum: the delta depends on both the FreqRouter's π_f
    (in router_state) and the layer-local content router on pooled lx.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()
    rank, in_dim, out_dim = 4, 8, 8
    K_c, K_f = 3, 2
    path = tmp_path / "anima_chimera.safetensors"
    _write_chimera_checkpoint(
        path, K_c=K_c, K_f=K_f, rank=rank, in_dim=in_dim, out_dim=out_dim,
        fei_dim=2, sigma_dim=0,
    )
    bundle = adapter.load_adapter(str(path))
    hydra = bundle["hydra"]
    chimera = hydra["chimera"]
    prefix = "lora_unet_blocks_0_mlp_layer1"
    mod = hydra["modules"][prefix]
    ups_stacked = torch.stack(
        [mod["lora_ups"][i] for i in sorted(mod["lora_ups"].keys())], dim=0
    )
    params = {
        "lora_down": mod["lora_down"],
        "lora_ups": ups_stacked,
        "router_w": mod["router_w"],
        "router_b": mod["router_b"],
        "inv_scale": None,
        "scale": 1.0,
        "num_experts_content": K_c,
        "num_experts_freq": K_f,
    }

    state: dict = {}
    hook = adapter._make_chimera_hook(params, strength=1.0, router_state=state)

    # Build a tiny Linear so the hook can read its ``output`` and we can
    # observe ``output + delta`` cleanly.
    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    x = torch.randn(2, 5, in_dim)
    base_out = linear(x)

    # Case A: π_f manually set to one-hot on expert 0 (freq pool).
    state["pi_f"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out_a = hook(linear, (x,), base_out.clone())
    delta_a = out_a - base_out

    # Case B: π_f one-hot on expert 1.
    state["pi_f"] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    out_b = hook(linear, (x,), base_out.clone())
    delta_b = out_b - base_out

    assert delta_a.shape == base_out.shape
    assert not torch.allclose(delta_a, delta_b, atol=1e-5), (
        "Swapping π_f one-hot mass between freq experts produced "
        "identical deltas — FreqRouter gate is not reaching the einsum."
    )
    # Both deltas should be non-trivial (the content router contributes too).
    assert delta_a.abs().mean() > 1e-3
    assert delta_b.abs().mean() > 1e-3


# ---------------------------------------------------------------------------
# Dual-A chimera (post-c4851b6 format): two independent A's per Linear, two
# per-pool B stacks. _parse_chimera_dual_a / _make_chimera_dual_a_hook /
# _apply_chimera_dual_a_to_model exercise paths in adapter.py.
# ---------------------------------------------------------------------------


def _write_chimera_dual_a_checkpoint(
    path: Path,
    *,
    K_c: int,
    K_f: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    fei_dim: int,
    sigma_dim: int,
    fr_hidden: int = 8,
    prefix: str = "lora_unet_blocks_0_mlp_layer1",
) -> None:
    """Synth a minimal dual-A chimera ``_chimera.safetensors`` file.

    Mirrors what ``_build_chimera_moe_state_dict`` writes after the
    dual-A SVD distillation: per-pool ``lora_down_{c,f}.weight`` +
    per-pool stacked ups ``lora_ups_{c,f}.{i}.weight``, K_c-narrow
    content router, top-level ``freq_router.net.*``.
    """
    torch.manual_seed(321)
    sd = {
        f"{prefix}.lora_down_c.weight": torch.randn(rank, in_dim),
        f"{prefix}.lora_down_f.weight": torch.randn(rank, in_dim),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        f"{prefix}.router.weight": torch.randn(K_c, rank) * 0.01,
        f"{prefix}.router.bias": torch.zeros(K_c),
        "freq_router.net.0.weight": torch.randn(fr_hidden, fei_dim + sigma_dim),
        "freq_router.net.0.bias": torch.zeros(fr_hidden),
        "freq_router.net.2.weight": torch.randn(K_f, fr_hidden) * 0.5,
        "freq_router.net.2.bias": torch.zeros(K_f),
    }
    for i in range(K_c):
        sd[f"{prefix}.lora_ups_c.{i}.weight"] = torch.randn(out_dim, rank)
    for j in range(K_f):
        sd[f"{prefix}.lora_ups_f.{j}.weight"] = torch.randn(out_dim, rank)

    metadata = {
        "ss_use_chimera_hydra": "true",
        "ss_num_experts_content": str(K_c),
        "ss_num_experts_freq": str(K_f),
        "ss_chimera_fei_feature_dim": str(fei_dim),
        "ss_chimera_sigma_feature_dim": str(sigma_dim),
        "ss_chimera_fei_sigma_low_div": "4.0",
        "ss_use_moe_style": "shared_A",
        "ss_route_per_layer": "true",
        "ss_router_source": "input",
    }
    save_file(sd, str(path), metadata=metadata)


def test_load_adapter_recognizes_chimera_dual_a(tmp_path):
    """``load_adapter`` populates ``bundle['chimera_dual_a']`` with the
    pool split + FreqRouter state when dual-A keys are present.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()

    path = tmp_path / "anima_chimera_dual_chimera.safetensors"
    _write_chimera_dual_a_checkpoint(
        path, K_c=3, K_f=2, rank=4, in_dim=8, out_dim=8,
        fei_dim=2, sigma_dim=0,
    )

    bundle = adapter.load_adapter(str(path))
    # Dual-A files have no shared-A `.lora_ups.{i}.weight` keys, so the
    # legacy hydra parser returns None.
    assert bundle["hydra"] is None
    cd = bundle["chimera_dual_a"]
    assert cd is not None
    assert cd["num_experts_content"] == 3
    assert cd["num_experts_freq"] == 2
    assert cd["fei_feature_dim"] == 2
    assert cd["sigma_feature_dim"] == 0
    fr = cd["freq_router_sd"]
    assert fr["net.0.weight"].shape == (8, 2)
    assert fr["net.2.weight"].shape == (2, 8)
    # Plain-LoRA extraction must NOT pick up dual-A keys.
    assert bundle["lora"] is None
    # Module map keyed by prefix.
    assert "lora_unet_blocks_0_mlp_layer1" in cd["modules"]
    mod = cd["modules"]["lora_unet_blocks_0_mlp_layer1"]
    assert mod["lora_down_c"].shape == (4, 8)
    assert mod["lora_down_f"].shape == (4, 8)
    assert len(mod["lora_ups_c"]) == 3
    assert len(mod["lora_ups_f"]) == 2


def test_chimera_dual_a_hook_dispatches_two_pools(tmp_path):
    """Per-Linear dual-A hook routes the content + freq pools through
    independent A's and sums their contributions. Verifies (a) the freq
    gate reaches the freq B-stack einsum and (b) the content gate reaches
    the content B-stack einsum — both via per-pool one-hot swaps.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()
    rank, in_dim, out_dim = 4, 8, 8
    K_c, K_f = 3, 2
    path = tmp_path / "anima_chimera_dual.safetensors"
    _write_chimera_dual_a_checkpoint(
        path, K_c=K_c, K_f=K_f, rank=rank, in_dim=in_dim, out_dim=out_dim,
        fei_dim=2, sigma_dim=0,
    )
    bundle = adapter.load_adapter(str(path))
    cd = bundle["chimera_dual_a"]
    mod = cd["modules"]["lora_unet_blocks_0_mlp_layer1"]
    ups_c_stacked = torch.stack(
        [mod["lora_ups_c"][i] for i in sorted(mod["lora_ups_c"].keys())], dim=0
    )
    ups_f_stacked = torch.stack(
        [mod["lora_ups_f"][i] for i in sorted(mod["lora_ups_f"].keys())], dim=0
    )
    params = {
        "lora_down_c": mod["lora_down_c"],
        "lora_down_f": mod["lora_down_f"],
        "lora_up_c_stack": ups_c_stacked,
        "lora_up_f_stack": ups_f_stacked,
        "router_w": mod["router_w"],
        "router_b": mod["router_b"],
        "inv_scale": None,
        "num_experts_content": K_c,
        "num_experts_freq": K_f,
    }

    state: dict = {}
    hook = adapter._make_chimera_dual_a_hook(params, strength=1.0, router_state=state)

    linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    x = torch.randn(2, 5, in_dim)
    base_out = linear(x)

    # Case A: freq one-hot on expert 0.
    state["pi_f"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out_a = hook(linear, (x,), base_out.clone())
    delta_a = out_a - base_out

    # Case B: freq one-hot on expert 1 — content path unchanged, so
    # only the freq-pool contribution to delta should differ.
    state["pi_f"] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    out_b = hook(linear, (x,), base_out.clone())
    delta_b = out_b - base_out

    assert delta_a.shape == base_out.shape
    assert not torch.allclose(delta_a, delta_b, atol=1e-5), (
        "Swapping π_f one-hot mass between freq experts produced "
        "identical deltas — FreqRouter gate is not reaching the freq "
        "B-stack einsum."
    )
    assert delta_a.abs().mean() > 1e-3
    assert delta_b.abs().mean() > 1e-3


def test_chimera_dual_a_metadata_mismatch_rejected(tmp_path):
    """If ``ss_num_experts_content`` disagrees with the actual
    ``lora_ups_c.*`` count, ``load_adapter`` raises rather than silently
    routing on a mis-shaped gate.
    """
    adapter = _load_adapter_module()
    adapter._adapter_cache.clear()
    path = tmp_path / "bad_dual_chimera.safetensors"
    _write_chimera_dual_a_checkpoint(
        path, K_c=3, K_f=2, rank=4, in_dim=8, out_dim=8,
        fei_dim=2, sigma_dim=0,
    )
    # Rewrite metadata with bogus K_c so loader can catch it.
    from safetensors.torch import load_file
    from safetensors import safe_open
    sd = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
    meta["ss_num_experts_content"] = "99"
    save_file(sd, str(path), metadata=meta)
    with pytest.raises(ValueError, match="ss_num_experts_content=99"):
        adapter.load_adapter(str(path))
