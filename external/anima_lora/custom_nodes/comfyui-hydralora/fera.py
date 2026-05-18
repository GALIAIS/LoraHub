"""FeRA (Frequency-Energy constrained Routing) loading for Anima.

Loader for two checkpoint formats with identical inference semantics
(global FEI router + per-Linear independent stacked experts):

  1. **Author-faithful FeRA** — ``networks.methods.fera``, the original
     port of Yin et al., arXiv:2511.17979. Stacked Parameters appear as
     ``lora_unet_*.lora_down`` / ``.lora_up`` (no ``.weight`` suffix),
     router under ``router.net.*``. N-band FEI with ``[high, ..., low]``
     ordering. Retired on the training side at plan2 but checkpoints
     still load here.

  2. **Plan2 stacked-experts** — ``networks.lora_anima`` with
     ``ss_network_spec=stacked_experts_global_fei`` (FeRA cell of the
     three-axis routing matrix: ``independent_A`` / ``route_per_layer=False``
     / ``router_source="fei"``). Per-expert split keys
     ``lora_unet_*.lora_downs.{i}.weight`` / ``.lora_ups.{i}.weight``,
     router under ``global_router.net.*``. 2-band FEI with
     ``[e_low, e_high]`` ordering (matches ``library/runtime/fei.py``).
     Saved by ``networks/lora_save.py::_build_stacked_experts_state_dict``
     as a ``*_moe.safetensors`` sibling.

Both formats share:
  * One **global router** consumes the latent's spectral energy and
    emits a single ``(B, num_experts)`` gate that every adapted Linear
    reuses for this step.
  * Each Linear carries **independent** stacked low-rank experts —
    ``lora_down: (E, r, in)`` / ``lora_up: (E, out, r)``.
  * Mutually exclusive with HydraLoRA-moe at the inference layer —
    ``library/inference/models.py`` refuses to load both. Same rule here.

Application strategy mirrors the training-side semantics:

  1. One ``forward_pre_hook`` on ``diffusion_model._forward_pre_hooks``
     computes the per-step Frequency-Energy Indicator from ``args[0]``
     (the latent) and runs the router, writing ``(B, num_experts)`` gates
     into shared state.
  2. One ``forward_hook`` per adapted Linear adds the gated stacked-expert
     correction to that Linear's output.

Both use ``ModelPatcher.add_object_patch`` on ``_forward_hooks`` /
``_forward_pre_hooks`` rather than overriding ``forward``. Overriding
``forward`` strands sub-Linears on CPU under ComfyUI's cast-weights path
(see CLAUDE.md). A hook leaves ``forward`` untouched and is properly
reverted on ``unpatch_model``.
"""

import logging
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F

# Both FEI paths live in library/inference/router_compute.py — the single
# source-of-truth re-exported by adapter.py after live-or-vendor resolution.
# Author-faithful FeRA uses ``compute_fei_nband_high_to_low`` (high-first
# ordering matching the retired ``networks/methods/fera.py::Frequency
# EnergyIndicator``); the plan2 ``stacked_experts_global_fei`` format uses
# ``compute_fei_2band`` (low-first, matching ``library/runtime/fei.py``).
# Trained router weights are bit-sensitive to band ordering — do not unify
# the two.
from .adapter import compute_fei_2band, compute_fei_nband_high_to_low

logger = logging.getLogger(__name__)

# Cache: path -> parsed bundle. Reuses adapter.py's pattern.
_fera_cache: Dict[str, dict] = {}


def _looks_like_fera_author(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Author-faithful FeRA (``networks.methods.fera``) key sniff.

    Pairs ``router.net.*`` MLP keys with stacked Parameter keys
    ``lora_unet_*.lora_down`` / ``.lora_up`` (no ``.weight`` suffix —
    flat Parameters, not ``nn.Linear`` children).
    """
    has_router = any(k.startswith("router.net.") for k in weights_sd)
    has_stacked = any(
        k.startswith("lora_unet_")
        and (k.endswith(".lora_down") or k.endswith(".lora_up"))
        for k in weights_sd
    )
    return has_router and has_stacked


def _looks_like_stacked_experts_global_fei(
    weights_sd: Dict[str, torch.Tensor],
) -> bool:
    """Plan2 ``stacked_experts_global_fei`` key sniff.

    Pairs ``global_router.net.*`` (network-level router on
    ``LoRANetwork``) with per-expert split keys
    ``lora_unet_*.lora_downs.{i}.weight`` / ``.lora_ups.{i}.weight``
    (independent-A stacked experts saved per expert + per fused
    q/k/v component by ``_build_stacked_experts_state_dict``).
    """
    has_router = any(k.startswith("global_router.net.") for k in weights_sd)
    has_split_downs = any(
        k.startswith("lora_unet_") and ".lora_downs." in k and k.endswith(".weight")
        for k in weights_sd
    )
    return has_router and has_split_downs


def _looks_like_fera(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Either FeRA variant — author-faithful or plan2 stacked_experts."""
    return _looks_like_fera_author(
        weights_sd
    ) or _looks_like_stacked_experts_global_fei(weights_sd)


def _parse_fera_author(
    weights_sd: Dict[str, torch.Tensor], meta: Dict[str, str], file_path: str
) -> dict:
    """Author-faithful ``networks.methods.fera`` format.

    Keys:
      * ``router.net.0/2.weight/bias`` — 2-layer router MLP.
      * ``lora_unet_*.lora_down`` / ``.lora_up`` — stacked Parameters
        (no ``.weight`` suffix), shape ``(E, r, in)`` / ``(E, out, r)``.

    Router input is N-band FEI with ``[high, ..., low]`` ordering
    (``compute_fei_nband_high_to_low``).
    """
    required = (
        "router.net.0.weight",
        "router.net.0.bias",
        "router.net.2.weight",
        "router.net.2.bias",
    )
    missing = [k for k in required if k not in weights_sd]
    if missing:
        raise ValueError(
            f"{file_path}: router keys missing ({missing}) — checkpoint "
            "may be from a non-faithful FeRA variant."
        )
    router = {
        "w1": weights_sd["router.net.0.weight"],
        "b1": weights_sd["router.net.0.bias"],
        "w2": weights_sd["router.net.2.weight"],
        "b2": weights_sd["router.net.2.bias"],
    }

    layers: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if not key.startswith("lora_unet_"):
            continue
        if key.endswith(".lora_down"):
            prefix = key[: -len(".lora_down")]
            layers.setdefault(prefix, {})["lora_down"] = value
        elif key.endswith(".lora_up"):
            prefix = key[: -len(".lora_up")]
            layers.setdefault(prefix, {})["lora_up"] = value

    incomplete = [
        p for p, d in layers.items() if "lora_down" not in d or "lora_up" not in d
    ]
    if incomplete:
        logger.warning(
            f"FeRA: {len(incomplete)} prefix(es) missing lora_down or "
            f"lora_up; skipping (first few: {incomplete[:5]})"
        )
        for p in incomplete:
            del layers[p]
    if not layers:
        raise ValueError(
            f"{file_path}: parsed router but no usable per-Linear experts."
        )

    sample_down = next(iter(layers.values()))["lora_down"]  # (E, r, in)
    E_shape, r_shape = int(sample_down.shape[0]), int(sample_down.shape[1])
    router_hidden, router_in = int(router["w1"].shape[0]), int(router["w1"].shape[1])

    def _mi(key: str, fb: int) -> int:
        v = meta.get(f"ss_{key}")
        try:
            return int(v) if v is not None else fb
        except (TypeError, ValueError):
            return fb

    def _mf(key: str, fb: float) -> float:
        v = meta.get(f"ss_{key}")
        try:
            return float(v) if v is not None else fb
        except (TypeError, ValueError):
            return fb

    cfg = {
        "rank": r_shape,
        "alpha": _mf("fera_alpha", float(r_shape)),
        "num_experts": E_shape,
        "num_bands": router_in,
        "router_tau": _mf("fera_router_tau", 0.7),
        "router_hidden": router_hidden,
        "fei_sigma_low_div": _mf("fei_sigma_low_div", 8.0),
        "fei_kind": "nband",
        "variant": "author",
    }
    cfg["scale"] = cfg["alpha"] / cfg["rank"]
    if _mi("fera_num_experts", E_shape) != E_shape:
        logger.warning(
            f"FeRA: ss_fera_num_experts mismatch with stacked axis ({E_shape}); using shape."
        )
    return {"router": router, "layers": layers, "cfg": cfg}


def _parse_stacked_experts_global_fei(
    weights_sd: Dict[str, torch.Tensor], meta: Dict[str, str], file_path: str
) -> dict:
    """Plan2 ``stacked_experts_global_fei`` format.

    Keys:
      * ``global_router.net.0/2.weight/bias`` — 2-layer router MLP
        (``LoRANetwork.GlobalRouter``).
      * ``lora_unet_*.lora_downs.{i}.weight`` / ``.lora_ups.{i}.weight`` —
        per-expert stacked into ``(E, r, in)`` / ``(E, out, r)``.
      * Per-fused-projection q/k/v are pre-split on disk (one prefix per
        Linear), so a direct ``named_modules()`` walk on cosmos backbones
        finds each prefix unchanged — same as the author-faithful path.

    Router input is **2-band FEI** with ``[e_low, e_high]`` ordering
    (``compute_fei_2band``, the single canonical 2-band impl shared with
    ``library/runtime/fei.py``). Plan2 default ``fei_sigma_low_div=4.0``.
    """
    required = (
        "global_router.net.0.weight",
        "global_router.net.0.bias",
        "global_router.net.2.weight",
        "global_router.net.2.bias",
    )
    missing = [k for k in required if k not in weights_sd]
    if missing:
        raise ValueError(
            f"{file_path}: global_router keys missing ({missing}) — "
            "checkpoint may be a partially-saved stacked_experts variant."
        )
    router = {
        "w1": weights_sd["global_router.net.0.weight"],
        "b1": weights_sd["global_router.net.0.bias"],
        "w2": weights_sd["global_router.net.2.weight"],
        "b2": weights_sd["global_router.net.2.bias"],
    }

    # Gather per-expert split shards: prefix -> {"lora_downs": {i: t},
    # "lora_ups": {i: t}, "alpha": t}.
    layers_raw: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if not key.startswith("lora_unet_"):
            continue
        # .lora_downs.{i}.weight
        if ".lora_downs." in key and key.endswith(".weight"):
            prefix = key.split(".lora_downs.")[0]
            idx = int(key.split(".lora_downs.")[1].split(".")[0])
            layers_raw.setdefault(prefix, {}).setdefault("downs", {})[idx] = value
        elif ".lora_ups." in key and key.endswith(".weight"):
            prefix = key.split(".lora_ups.")[0]
            idx = int(key.split(".lora_ups.")[1].split(".")[0])
            layers_raw.setdefault(prefix, {}).setdefault("ups", {})[idx] = value
        elif key.endswith(".alpha") and key.count(".") == 1:
            # Bare ``lora_unet_X.alpha`` (no expert index). Module-level
            # shared alpha matches save format.
            prefix = key[: -len(".alpha")]
            layers_raw.setdefault(prefix, {})["alpha"] = value

    layers: Dict[str, dict] = {}
    incomplete: List[str] = []
    for prefix, parts in layers_raw.items():
        downs = parts.get("downs") or {}
        ups = parts.get("ups") or {}
        if not downs or not ups:
            incomplete.append(prefix)
            continue
        if sorted(downs.keys()) != sorted(ups.keys()):
            incomplete.append(prefix)
            continue
        idxs = sorted(downs.keys())
        try:
            lora_down = torch.stack([downs[i] for i in idxs], dim=0)
            lora_up = torch.stack([ups[i] for i in idxs], dim=0)
        except RuntimeError as exc:
            logger.warning(
                f"FeRA(stacked_experts): {prefix} expert shape mismatch: {exc}"
            )
            incomplete.append(prefix)
            continue
        entry = {"lora_down": lora_down, "lora_up": lora_up}
        if "alpha" in parts:
            entry["alpha"] = parts["alpha"]
        layers[prefix] = entry

    if incomplete:
        logger.warning(
            f"FeRA(stacked_experts): {len(incomplete)} prefix(es) skipped "
            f"(incomplete or mismatched experts; first few: {incomplete[:5]})"
        )
    if not layers:
        raise ValueError(
            f"{file_path}: parsed global_router but no usable per-Linear experts."
        )

    sample = next(iter(layers.values()))
    E_shape, r_shape = (
        int(sample["lora_down"].shape[0]),
        int(sample["lora_down"].shape[1]),
    )
    router_hidden = int(router["w1"].shape[0])
    router_in = int(router["w1"].shape[1])  # = fei_feature_dim

    def _mf(key: str, fb: float) -> float:
        v = meta.get(f"ss_{key}")
        try:
            return float(v) if v is not None else fb
        except (TypeError, ValueError):
            return fb

    # alpha may live per-prefix or fall back to the metadata-stamped
    # ss_network_alpha. Sample's per-prefix alpha wins when present.
    sample_alpha_t = sample.get("alpha")
    if sample_alpha_t is not None:
        alpha_default = float(
            sample_alpha_t.item() if hasattr(sample_alpha_t, "item") else sample_alpha_t
        )
    else:
        alpha_default = _mf("network_alpha", float(r_shape))

    # Plan2 defaults: tau=0.7, fei_sigma_low_div=4.0. Neither stamped by
    # default in the save metadata today, so we just hard-default.
    cfg = {
        "rank": r_shape,
        "alpha": alpha_default,
        "num_experts": E_shape,
        "num_bands": router_in,  # = fei_feature_dim, expected 2
        "router_tau": _mf("router_tau", 0.7),
        "router_hidden": router_hidden,
        "fei_sigma_low_div": _mf("fei_sigma_low_div", 4.0),
        "fei_kind": "2band_low_high",
        "variant": "stacked_experts_global_fei",
    }
    cfg["scale"] = cfg["alpha"] / cfg["rank"]
    if router_in != 2:
        logger.warning(
            f"FeRA(stacked_experts): router input dim {router_in} ≠ 2 — "
            "expected 2-band FEI. Treating as opaque routing key (will "
            "compute 2 bands and slice/pad to fit, gate may misbehave)."
        )
    return {"router": router, "layers": layers, "cfg": cfg}


def load_fera(file_path: str) -> dict:
    """Parse a FeRA checkpoint once, cache by path.

    Auto-routes between author-faithful (``networks.methods.fera``) and
    plan2 stacked-experts (``stacked_experts_global_fei``) formats.
    Returns a bundle with router MLP weights, per-Linear stacked expert
    weights keyed by ``lora_unet_<dotted>`` prefix, and a config dict
    (rank, alpha, num_experts, num_bands, router_tau, router_hidden,
    fei_sigma_low_div, fei_kind, variant, scale).
    """
    if file_path in _fera_cache:
        return _fera_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)
    with safe_open(file_path, framework="pt") as f:
        meta = dict(f.metadata() or {})

    network_module = meta.get("ss_network_module", "")
    network_spec = meta.get("ss_network_spec", "")

    # Discriminate by metadata first (cheaper, unambiguous), fall back
    # to a key sniff. Either format declares its variant explicitly on
    # current training runs.
    is_stacked = (
        network_spec == "stacked_experts_global_fei"
        or _looks_like_stacked_experts_global_fei(weights_sd)
    )
    is_author = not is_stacked and (
        network_module == "networks.methods.fera" or _looks_like_fera_author(weights_sd)
    )

    if is_stacked:
        parsed = _parse_stacked_experts_global_fei(weights_sd, meta, file_path)
    elif is_author:
        parsed = _parse_fera_author(weights_sd, meta, file_path)
    else:
        raise ValueError(
            f"{file_path} doesn't look like a FeRA checkpoint "
            "(no router.net.* + lora_unet_*.lora_down/lora_up, and no "
            "global_router.net.* + lora_unet_*.lora_downs.{i}.weight). "
            "For LoRA / HydraLoRA / ReFT files, use AnimaAdapterLoader."
        )

    bundle = {
        "path": file_path,
        "router": parsed["router"],
        "layers": parsed["layers"],
        "cfg": parsed["cfg"],
    }
    _fera_cache[file_path] = bundle

    cfg = bundle["cfg"]
    logger.info(
        f"Loaded FeRA[{cfg['variant']}]: {len(bundle['layers'])} adapted Linears, "
        f"{cfg['num_experts']} experts × rank {cfg['rank']}, "
        f"router({cfg['num_bands']} bands → {cfg['router_hidden']} → "
        f"{cfg['num_experts']}, τ={cfg['router_tau']:.2f}, "
        f"fei_kind={cfg['fei_kind']}), σ_low_div={cfg['fei_sigma_low_div']:g} "
        f"from {file_path}"
    )
    return bundle


def _make_fera_pre_hook(router: dict, cfg: dict, fera_state: dict):
    """Forward pre-hook that runs the global router on the current latent.

    Writes ``fera_state["gates"]`` of shape ``(B, num_experts)`` once per
    ``diffusion_model`` forward. Per-Linear hooks read from there. Router
    weights migrate to the latent's device + fp32 on first call and stay
    there for the rest of the session.

    Dispatches the FEI compute by ``cfg["fei_kind"]``:
      * ``"nband"`` (author-faithful) — N-band pyramid, ``[high, ..., low]``.
      * ``"2band_low_high"`` (plan2 stacked-experts) — 2-band Laplacian,
        ``[e_low, e_high]``, matches ``library/runtime/fei.py``.

    The ``@torch._dynamo.disable`` guard mirrors
    ``adapter.py::_make_router_pre_hook`` — the dict store + FEI conv2d
    shouldn't get inlined into the compiled DiT graph or every per-Linear
    cast logs a DeviceCopy warning per step.
    """
    state = {
        "w1": router["w1"],
        "b1": router["b1"],
        "w2": router["w2"],
        "b2": router["b2"],
        "device": None,
    }
    tau = float(cfg["router_tau"])
    num_bands = int(cfg["num_bands"])
    fei_sigma_low_div = float(cfg["fei_sigma_low_div"])
    fei_kind = str(cfg.get("fei_kind", "nband"))
    router_in = int(state["w1"].shape[1])

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        for k in ("w1", "b1", "w2", "b2"):
            state[k] = state[k].to(device=x.device, dtype=torch.float32)
        state["device"] = x.device

    @torch._dynamo.disable
    def fera_pre_hook(module, args):
        if len(args) < 1 or args[0] is None:
            return
        x = args[0].detach()
        _ensure_on_device(x)
        # Anima/cosmos passes a 5D (B, C, T, H, W) latent; collapse T=1
        # so the 2D Laplacian sees (B, C, H, W). Other backbones pass 4D
        # through unchanged.
        if x.dim() == 5:
            x = x.squeeze(2)
        h_lat, w_lat = int(x.shape[-2]), int(x.shape[-1])
        sigma_low = float(min(h_lat, w_lat)) / fei_sigma_low_div
        if fei_kind == "2band_low_high":
            fei = compute_fei_2band(x, sigma_low)
            # Trim/pad to the router's input width — defensive against
            # off-spec checkpoints with router_in != 2.
            if fei.shape[-1] > router_in:
                fei = fei[..., :router_in]
            elif fei.shape[-1] < router_in:
                fei = F.pad(fei, (0, router_in - fei.shape[-1]))
        else:
            fei = compute_fei_nband_high_to_low(x, sigma_low, num_bands)
        hidden = F.relu(F.linear(fei, state["w1"], state["b1"]))
        logits = F.linear(hidden, state["w2"], state["b2"])
        fera_state["gates"] = F.softmax(logits / tau, dim=-1)

    return fera_pre_hook


def _make_fera_clear_hook(fera_state: dict):
    """Forward (post) hook that drops gates at the end of each
    ``diffusion_model.forward``.

    Defense-in-depth against stale gates leaking into code paths that
    run adapted Linears *outside* the diffusion forward — e.g. ComfyUI's
    ``BaseModel.extra_conds`` calls ``diffusion_model.preprocess_text_embeds``
    before sampling starts, and on a re-run the previous sample's
    last-step gates (shape ``(B_latent, E)`` with B_latent doubled by CFG)
    would otherwise broadcast through any FeRA hook fired in that
    pre-sample path. The LLM adapter is excluded from the key map for
    exactly this reason, but this clear keeps the invariant
    independent of which modules end up adapted.
    """

    @torch._dynamo.disable
    def fera_clear_hook(module, inputs, output):
        fera_state["gates"] = None
        return output

    return fera_clear_hook


def _make_fera_hook(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: float,
    strength: float,
    fera_state: dict,
):
    """Per-Linear forward hook that adds ``Σ_k w_k · U_k @ D_k @ x``.

    Mirrors ``FeRALinear.forward`` in ``networks/methods/fera.py``: one
    batched ``einsum`` for the down projection over E experts, multiply
    by the per-batch gates, one batched ``einsum`` for the up projection.
    Weight tensors migrate to the input's device + fp32 on first call;
    subsequent calls skip the migration. Per-Linear hot path stays in
    fp32 to match the CLI's precision policy (also matches
    ``_make_hydra_hook``).

    Returns the original output untouched when ``strength=0`` or
    ``gates`` hasn't been written yet (defensive — the pre-hook should
    have fired in the same forward).
    """
    state = {
        "lora_down": lora_down,  # (E, r, in)
        "lora_up": lora_up,  # (E, out, r)
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        state["lora_down"] = state["lora_down"].to(device=x.device, dtype=torch.float32)
        state["lora_up"] = state["lora_up"].to(device=x.device, dtype=torch.float32)
        state["device"] = x.device

    def fera_hook(module, inputs, output):
        if strength == 0.0:
            return output
        gates = fera_state.get("gates")
        if gates is None:
            return output

        x = inputs[0]
        _ensure_on_device(x)
        x_c = x.float()

        # (..., in) × (E, r, in)ᵀ → (..., E, r). Stacked-einsum saves
        # one (..., E, r) activation instead of E × (..., D_out); not as
        # impactful at inference (no backward) but the layout matches
        # the training code so semantics are bit-identical.
        lx = torch.einsum("...i,eri->...er", x_c, state["lora_down"])

        # Broadcast gates (B, E) across any mid dims (e.g. token T).
        B, E = gates.shape
        n_mid = lx.ndim - 3  # dims between batch and (E, r)
        view_shape = (B,) + (1,) * n_mid + (E, 1)
        lx = lx * gates.view(view_shape).to(torch.float32)

        # (..., E, r) × (E, out, r)ᵀ → (..., out)
        delta = torch.einsum("...er,eor->...o", lx, state["lora_up"])
        return output + (delta * (scale * strength)).to(output.dtype)

    return fera_hook


def _resolve_module(model, dotted_path: str):
    """Walk attribute / index path under ``model.model``. Same idiom as
    ``adapter.py::_resolve_module``."""
    obj = model.model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _build_fera_key_map(model) -> Dict[str, str]:
    """Inverse-map FeRA's ``lora_unet_<...>`` prefixes to live module paths
    under ``model.model``.

    Built from a direct walk of ``diffusion_model.named_modules()`` rather
    than ``comfy.lora.model_lora_keys_unet`` because ComfyUI's helper
    doesn't emit keys for fused projections (cosmos ``qkv_proj`` /
    ``kv_proj``) — it was designed around the split q/k/v convention from
    older UNet checkpoints. FeRA's ``_scan_targets`` walks
    ``named_modules()`` directly on the training side and matches the
    fused names through its target regex, so a checkpoint trained on the
    full default target set (attn + MLP) has 2 prefixes per block that
    aren't in ComfyUI's map.

    The FeRA training-side prefix convention is

        lora_name = "lora_unet_" + dotted_path.replace(".", "_")

    so this map is the literal inverse, built once at apply time. No
    ambiguity in the inverse direction — each live Linear produces
    exactly one ``lora_name``.

    ``llm_adapter.*`` Linears are intentionally excluded. The LLM adapter
    runs *outside* ``diffusion_model.forward`` at ComfyUI inference time —
    ``model_base.Anima.extra_conds`` calls ``preprocess_text_embeds``
    directly before the sampling loop, so the router pre-hook (which is
    on ``diffusion_model._forward_pre_hooks``) has not fired yet and
    ``fera_state["gates"]`` is either ``None`` (first sample) or *stale*
    from the previous sample's last step. Stale gates from a CFG-doubled
    latent (B=2) broadcast the per-Linear delta to B=2 in a B=1 text
    forward, blowing up ``q_proj``'s output to twice the expected size
    and crashing ``view([B, T, n_heads, head_dim])`` in the next line of
    ``Attention.forward``. Training-side FeRA contributions on
    ``llm_adapter.*_q_proj`` therefore can't be replayed at ComfyUI
    inference — drop them on the floor here. The DiT (cosmos backbone)
    FeRA contributions still apply correctly because those Linears run
    inside ``diffusion_model.forward`` *after* the router pre-hook fires.
    """
    import torch.nn as nn

    diffusion = model.get_model_object("diffusion_model")
    out: Dict[str, str] = {}
    for name, child in diffusion.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Strip torch.compile wrapper if any — matches
        # FeRANetwork._scan_targets so the lora_name keys agree.
        clean = name.replace("_orig_mod.", "")
        if not clean:
            continue
        if clean.startswith("llm_adapter.") or clean == "llm_adapter":
            continue
        lora_name = "lora_unet_" + clean.replace(".", "_")
        out[lora_name] = f"diffusion_model.{clean}"
    return out


def apply_fera(model, file_path: str, strength: float) -> bool:
    """Apply a FeRA adapter to ``model`` in place. ``model`` must already
    be a clone. Returns True if at least one hook was installed.
    """
    if strength == 0:
        logger.info("FeRA: strength=0 — installing no hooks.")
        return False

    bundle = load_fera(file_path)
    layers = bundle["layers"]
    cfg = bundle["cfg"]

    # Per-checkpoint shared state. Pre-hook writes "gates", every
    # per-Linear hook reads from this dict by closure capture.
    fera_state: dict = {}

    # Install the model-level pre-hook (router runs once per forward).
    # Patch _forward_pre_hooks (an OrderedDict) via add_object_patch so
    # it's reverted on ModelPatcher.unpatch_model. Composes with any
    # prior diffusion_model.forward object_patch (postfix wraps forward;
    # the pre-hook fires before that wrapper sees args).
    diffusion_model = model.get_model_object("diffusion_model")
    pre_hook = _make_fera_pre_hook(bundle["router"], cfg, fera_state)
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(pre_hook)] = pre_hook
    model.add_object_patch("diffusion_model._forward_pre_hooks", new_pre_hooks)

    # Post-hook clears gates at the end of every diffusion forward so
    # subsequent samples don't read stale CFG-batched gates from the
    # previous run's last step (see _make_fera_clear_hook).
    clear_hook = _make_fera_clear_hook(fera_state)
    new_dm_post_hooks = OrderedDict(diffusion_model._forward_hooks)
    new_dm_post_hooks[id(clear_hook)] = clear_hook
    model.add_object_patch("diffusion_model._forward_hooks", new_dm_post_hooks)

    # Direct walk of diffusion_model — covers fused qkv/kv projections
    # that ComfyUI's model_lora_keys_unet doesn't enumerate. See
    # ``_build_fera_key_map`` for why.
    key_map = _build_fera_key_map(model)
    default_scale = float(cfg["scale"])
    rank = int(cfg["rank"])

    patched = 0
    skipped: list[str] = []
    for prefix, layer in layers.items():
        module_path = key_map.get(prefix)
        if module_path is None:
            skipped.append(f"{prefix}: no matching Linear under diffusion_model")
            continue
        try:
            linear = _resolve_module(model, module_path)
        except (AttributeError, IndexError, ValueError) as e:
            skipped.append(f"{prefix}: resolve {module_path} failed ({e})")
            continue
        # Per-prefix alpha wins over the bundle default — the stacked-
        # experts save path writes one ``.alpha`` per fused-projection
        # component, and modules can in principle have heterogeneous
        # alpha. Fall back to the bundle default when absent.
        alpha_t = layer.get("alpha")
        if alpha_t is not None:
            alpha_v = float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
            layer_scale = alpha_v / rank
        else:
            layer_scale = default_scale
        hook = _make_fera_hook(
            layer["lora_down"], layer["lora_up"], layer_scale, strength, fera_state
        )
        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"FeRA: skipped {len(skipped)} prefix(es); first few: {skipped[:5]}"
        )
    logger.info(
        f"FeRA[{cfg['variant']}]: installed router pre-hook + {patched} "
        f"per-Linear hooks (strength={strength}, {cfg['num_experts']} "
        f"experts × rank {cfg['rank']}, {cfg['num_bands']} bands, "
        f"fei_kind={cfg['fei_kind']}, σ_low_div={cfg['fei_sigma_low_div']:g})"
    )
    return patched > 0
