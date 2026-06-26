"""Exponential moving average for LoRA-network parameters.

The trainer applies LoRA-style adapters to a frozen DiT. We don't EMA
the DiT (it doesn't update); we EMA only the trainable adapter
parameters — i.e. the output of ``network.get_trainable_params()``.

Design:

* Shadow params live as a flat list aligned 1:1 with the iteration
  order of ``network.get_trainable_params()`` *at construction time*.
  We capture an ordered list of names too so checkpoint round-trips
  don't depend on dict ordering.
* ``step()`` is called once per optimizer step (post-clip,
  post-optimizer). With Diffusers-style warmup the effective decay
  is ``min(decay, (1 + n) / (10 + n))`` — gentler at the start of
  training where parameters are moving fast.
* ``swap()`` is a context manager that temporarily writes shadow
  values into the live network, runs whatever the caller wants
  (typically: build a state-dict, save), and restores. Used by the
  saver to write ``<ckpt>_ema.safetensors`` next to the live ckpt
  without permanently mutating training state.
* ``state_dict()`` / ``load_state_dict()`` round-trip via the captured
  name list so a fresh EMA instance built against a re-attached
  network on resume picks up the right shadows even if module
  registration order shifted slightly across versions.

Inspired by diffusers' ``EMAModel`` but trimmed to what the LoRA
trainer needs (no foreach kernels, no power-decay schedule, no
``model_cls`` round-trip).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _named_trainables(network: Any) -> list[tuple[str, torch.nn.Parameter]]:
    """Stable ordered list of trainable params from a LoRA network.

    ``network.get_trainable_params()`` is the canonical source — that's
    what the optimizer + clip path uses, so EMA must mirror its order
    exactly. We walk it once and pair each tensor with the matching
    ``named_parameters()`` name so checkpoints stay portable.
    """
    by_id = {id(p): name for name, p in network.named_parameters()}
    result: list[tuple[str, torch.nn.Parameter]] = []
    for i, p in enumerate(network.get_trainable_params()):
        if not isinstance(p, torch.nn.Parameter):
            # Some adapters return raw tensors. Skip — EMA only tracks
            # registered Parameters so the shadow dict stays restorable
            # via ``load_state_dict``.
            continue
        if not p.requires_grad:
            continue
        name = by_id.get(id(p))
        if name is None:
            if not getattr(network, "is_full_finetune", False):
                # Anonymous tensor (typically a fused view). EMA over views
                # would double-count the underlying storage, so skip.
                continue
            # Full-model finetune shims keep the DiT as an external
            # reference to avoid accelerator double-wrapping. Those
            # Parameters are not registered on the shim, but they are
            # still the optimizer's canonical trainables, so track them
            # by stable optimizer order.
            name = f"trainable_{i}"
        result.append((name, p))
    return result


class EMAModel:
    """Track an EMA of the LoRA network's trainable parameters.

    Args:
        network: the LoRA adapter (post-``accelerator.prepare``-unwrap).
        decay: target decay rate (e.g. 0.9999).
        use_num_updates: enable Diffusers-style warmup
            ``decay_eff = min(decay, (1 + n) / (10 + n))``.
        device: ``"gpu"`` keeps shadow params on the same device as the
            live params (fast updates, costs VRAM); ``"cpu"`` halves
            VRAM cost but every step pays a host-device copy.
    """

    def __init__(
        self,
        network: Any,
        *,
        decay: float = 0.9999,
        use_num_updates: bool = False,
        device: str = "gpu",
    ) -> None:
        if not 0.0 < decay < 1.0:
            msg = f"EMA decay must be in (0,1), got {decay}"
            raise ValueError(msg)
        self.decay = float(decay)
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0
        self.device_pref = device

        named = _named_trainables(network)
        self.param_names: list[str] = [n for n, _ in named]
        # Detached float32 clones — keeps EMA stable when the live
        # params are bf16/fp16 (otherwise (1-decay)*delta vanishes
        # under the lower mantissa).
        self.shadow_params: list[torch.Tensor] = []
        for _, p in named:
            shadow = p.detach().clone().to(torch.float32)
            if device == "cpu":
                shadow = shadow.to("cpu")
            self.shadow_params.append(shadow)

    @property
    def n_params(self) -> int:
        return len(self.shadow_params)

    def _effective_decay(self) -> float:
        if not self.use_num_updates:
            return self.decay
        # Diffusers / Karras et al. schedule: gentle at startup so the
        # first few hundred steps don't lock in noise.
        warmup = (1 + self.num_updates) / (10 + self.num_updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def step(self, network: Any) -> None:
        """Update shadow params from the live network.

        Robust to a network that grew/shrunk trainable params since
        construction (logs once, then bails — better than crashing
        mid-training). Honours ``num_updates`` warmup if enabled.
        """
        named = _named_trainables(network)
        if len(named) != len(self.shadow_params):
            logger.warning(
                "EMA: trainable param count drifted (%d -> %d), skipping update",
                len(self.shadow_params),
                len(named),
            )
            return

        decay = self._effective_decay()
        one_minus = 1.0 - decay
        for shadow, (_, live) in zip(self.shadow_params, named):
            live_data = live.detach().to(shadow.device, dtype=shadow.dtype)
            shadow.mul_(decay).add_(live_data, alpha=one_minus)
        self.num_updates += 1

    @contextlib.contextmanager
    def swap(self, network: Any) -> Iterator[None]:
        """Temporarily replace live params with shadow values.

        Use case: build a state-dict for saving without permanently
        mutating training state. The original tensors are restored
        on exit even if the body raises.
        """
        named = _named_trainables(network)
        if len(named) != len(self.shadow_params):
            logger.warning("EMA swap skipped: param count mismatch")
            yield
            return
        backups: list[torch.Tensor] = []
        try:
            for shadow, (_, live) in zip(self.shadow_params, named):
                backups.append(live.detach().clone())
                live.data.copy_(shadow.to(live.device, dtype=live.dtype))
            yield
        finally:
            for backup, (_, live) in zip(backups, named):
                live.data.copy_(backup)

    def to(self, device: str | torch.device) -> None:
        """Move shadow params (e.g. when reattaching after a load_state)."""
        for i, t in enumerate(self.shadow_params):
            self.shadow_params[i] = t.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "use_num_updates": self.use_num_updates,
            "num_updates": self.num_updates,
            "param_names": list(self.param_names),
            "shadow_params": [t.cpu() for t in self.shadow_params],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.use_num_updates = bool(state.get("use_num_updates", self.use_num_updates))
        self.num_updates = int(state.get("num_updates", 0))
        names = list(state.get("param_names") or [])
        shadows = list(state.get("shadow_params") or [])
        if names and shadows:
            self.param_names = names
            self.shadow_params = [t.detach().to(torch.float32) for t in shadows]
            if self.device_pref == "cpu":
                self.to("cpu")


__all__ = ["EMAModel", "_named_trainables"]
