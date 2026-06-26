"""Full DiT finetuning adapter for Anima.

This is intentionally a tiny trainer-facing shim, not a new trainer.  The
existing train.py loop expects a ``network_module`` object with LoRA-like
lifecycle hooks; this module exposes those hooks while training the base DiT
parameters directly and saving a complete Anima checkpoint.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from library.anima import weights as anima_weights

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del multiplier, network_dim, network_alpha, vae, neuron_dropout, kwargs
    return FullFinetuneNetwork(text_encoders=text_encoders, unet=unet)


def create_network_from_weights(
    multiplier: float,
    weights_file: str,
    vae,
    text_encoder,
    unet,
    for_inference: bool = False,
    **kwargs,
):
    del multiplier, weights_file, vae, text_encoder, unet, for_inference, kwargs
    raise NotImplementedError("full_finetune checkpoints are full Anima models, not adapters")


class FullFinetuneNetwork(nn.Module):
    network_module = "networks.methods.full_finetune"
    network_spec = "full_finetune"
    is_full_finetune = True

    def __init__(self, text_encoders: list, unet) -> None:
        super().__init__()
        self.text_encoders_ref = _as_list(text_encoders)
        self.unet_ref = [unet]
        self.train_text_encoder = False
        self.train_unet = True

    def set_multiplier(self, multiplier: float) -> None:
        del multiplier

    def is_mergeable(self) -> bool:
        return False

    def enable_gradient_checkpointing(self) -> None:
        pass

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        self.text_encoders_ref = _as_list(text_encoder)
        self.unet_ref[0] = unet
        self.train_text_encoder = bool(apply_text_encoder)
        self.train_unet = bool(apply_unet)
        unet.requires_grad_(self.train_unet)
        for enc in self.text_encoders_ref:
            enc.requires_grad_(self.train_text_encoder)
        logger.info(
            "FullFinetuneNetwork: train_unet=%s train_text_encoder=%s",
            self.train_unet,
            self.train_text_encoder,
        )

    def prepare_network(self, args) -> None:
        del args

    def prepare_grad_etc(self, text_encoder, unet) -> None:
        self.text_encoders_ref = _as_list(text_encoder)
        self.unet_ref[0] = unet
        unet.requires_grad_(self.train_unet)
        for enc in self.text_encoders_ref:
            enc.requires_grad_(self.train_text_encoder)

    def on_epoch_start(self, text_encoder, unet) -> None:
        del text_encoder
        unet.train(self.train_unet)
        for enc in self.text_encoders_ref:
            enc.train(self.train_text_encoder)

    def get_trainable_params(self):
        params = []
        unet = self.unet_ref[0]
        if self.train_unet:
            params.extend(p for p in unet.parameters() if p.requires_grad)
        if self.train_text_encoder:
            for enc in self.text_encoders_ref:
                params.extend(p for p in enc.parameters() if p.requires_grad)
        return params

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        groups = []
        descriptions = []
        base_lr = default_lr or unet_lr or text_encoder_lr
        if self.train_unet:
            unet_params = [p for p in self.unet_ref[0].parameters() if p.requires_grad]
            groups.append({"params": unet_params, "lr": unet_lr or base_lr})
            descriptions.append("anima_dit")
        if self.train_text_encoder:
            te_params = []
            for enc in self.text_encoders_ref:
                te_params.extend(p for p in enc.parameters() if p.requires_grad)
            if te_params:
                groups.append({"params": te_params, "lr": text_encoder_lr or base_lr})
                descriptions.append("text_encoder")
        return groups, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        params, _ = self.prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        return params

    def save_weights(self, file, dtype, metadata) -> None:
        meta = dict(metadata or {})
        meta["ss_network_module"] = self.network_module
        meta["ss_network_spec"] = self.network_spec
        meta["ss_full_finetune"] = "true"
        unet = _unwrap_model(self.unet_ref[0])
        anima_weights.save_anima_model(str(file), unet.state_dict(), meta, dtype)

    def load_weights(self, file):
        raise NotImplementedError("load the full Anima checkpoint as baseModel.checkpoint")
