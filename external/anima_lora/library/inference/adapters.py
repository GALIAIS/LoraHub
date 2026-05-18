"""Inference-time adapter state helpers."""

from collections.abc import Iterable
from typing import Any, Optional

import torch

from library.runtime.fei import compute_fei_2band, fei_sigma_low


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def iter_hydra_networks(model: Any) -> Iterable[Any]:
    """Yield attached HydraLoRA networks without duplicating aliases."""
    candidates = []
    containers = [model]
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None and orig_mod is not model:
        containers.append(orig_mod)

    for container in containers:
        candidates.extend(_as_iterable(getattr(container, "_hydra_networks", None)))
        candidates.extend(_as_iterable(getattr(container, "_hydra_network", None)))

        # Hydra inference also aliases the same network into the P-GRAFT slot for
        # cutoff support. Keep this fallback for older call sites, but only accept
        # routing-aware networks (σ or FEI) so regular P-GRAFT LoRAs remain untouched.
        pgraft_network = getattr(container, "_pgraft_network", None)
        if getattr(pgraft_network, "use_sigma_router", False) or getattr(
            pgraft_network, "use_fei_router", False
        ):
            candidates.append(pgraft_network)

    seen = set()
    for network in candidates:
        if network is None:
            continue
        ident = id(network)
        if ident in seen:
            continue
        seen.add(ident)
        yield network


def set_hydra_sigma(model: Any, timesteps: torch.Tensor) -> None:
    """Propagate current denoising sigma to router-live HydraLoRA adapters."""
    sigma = timesteps.detach().flatten().to(dtype=torch.float32)
    for network in iter_hydra_networks(model):
        set_sigma = getattr(network, "set_sigma", None)
        if callable(set_sigma):
            set_sigma(sigma)


def clear_hydra_sigma(model: Any) -> None:
    """Clear cached sigma from router-live HydraLoRA adapters."""
    for network in iter_hydra_networks(model):
        clear_sigma = getattr(network, "clear_sigma", None)
        if callable(clear_sigma):
            clear_sigma()


def set_hydra_fei(model: Any, fei: torch.Tensor) -> None:
    """Propagate per-sample FEI ``[B, 2]`` to router-live HydraLoRA adapters.

    Parallel to ``set_hydra_sigma``. ``fei`` is ``(B, fei_feature_dim)`` —
    the simplex energy from ``library.runtime.fei.compute_fei_2band``.
    Networks without a FEI router silently ignore the call (no
    ``set_fei`` attribute).
    """
    fei = fei.detach().to(dtype=torch.float32)
    for network in iter_hydra_networks(model):
        set_fei = getattr(network, "set_fei", None)
        if callable(set_fei):
            set_fei(fei)


def clear_hydra_fei(model: Any) -> None:
    """Clear cached FEI from router-live HydraLoRA adapters."""
    for network in iter_hydra_networks(model):
        clear_fei = getattr(network, "clear_fei", None)
        if callable(clear_fei):
            clear_fei()


def _resolve_fei_sigma_low_div(model: Any) -> Optional[float]:
    """First fei-aware network's ``cfg.fei_sigma_low_div``, or None if no
    FEI router is attached.

    Used by ``compute_and_set_hydra_fei`` to pick σ_low without the caller
    having to thread it through. All currently-shipped variants share one
    σ_low_div across the whole network.
    """
    for network in iter_hydra_networks(model):
        if not getattr(network, "use_fei_router", False):
            continue
        cfg = getattr(network, "cfg", None)
        if cfg is None:
            continue
        return float(getattr(cfg, "fei_sigma_low_div", 8.0))
    return None


def compute_and_set_hydra_fei(model: Any, z: torch.Tensor) -> None:
    """One-shot per-step FEI compute + propagate.

    ``z`` may be a 4D ``(B, C, H, W)`` latent or Anima's 5D
    ``(B, C, T, H, W)`` (with ``T == 1``) — the singleton temporal dim is
    squeezed automatically. No-op when no attached network has a FEI router
    (so non-FEI variants pay zero overhead beyond the one ``getattr`` per
    network).
    """
    div = _resolve_fei_sigma_low_div(model)
    if div is None:
        return
    if z.dim() == 5:
        z = z.squeeze(2)
    h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
    sigma_low = fei_sigma_low(h_lat, w_lat, div)
    fei = compute_fei_2band(z, sigma_low)
    set_hydra_fei(model, fei)


