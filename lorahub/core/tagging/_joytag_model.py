"""Vendored inference-only port of fancyfeast/joytag's ``Models.ViT``.

This file is a minimal, inference-focused port of the ``ViT`` class (and its
small helper modules) defined in fancyfeast's upstream ``Models.py``. The
upstream module is Apache-2.0 licensed and ships alongside the safetensors
weights in https://huggingface.co/fancyfeast/joytag. We vendor a trimmed copy
so LoraHub doesn't have to take a hard dependency on ``timm``, ``einops``,
``torchvision``, or ``transformers`` (none of those are actually needed for
the JoyTag config — its ViT is a tiny hand-rolled implementation).

What's intentionally left out vs upstream:
- training-only paths (``return_loss``/``calculate_loss``, ``patch_dropout``)
- the ``CLIPLikeModel`` / ``MaskedAutoEncoderViT`` siblings
- the loss-type dispatcher and pos-weight plumbing
- the abstract ``VisionModel`` base + ``from_config`` / ``load_model`` factory

The ``state_dict`` parameter names match upstream exactly so the safetensors
checkpoint at ``fancyfeast/joytag/model.safetensors`` loads via
``model.load_state_dict(...)`` without any key remapping.

Source: fancyfeast/joytag — Models.py (Apache-2.0).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def sinusoidal_position_embedding(
    width: int,
    height: int,
    depth: int,
    dtype: torch.dtype,
    device: torch.device,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """2-D sinusoidal positional embedding, flattened to ``(h * w, d)``."""
    if depth % 4 != 0:
        msg = "embedding dimension must be divisible by 4"
        raise ValueError(msg)

    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    omega = torch.arange(depth // 4, device=device) / (depth // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    embedding = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return embedding.type(dtype)


class StochDepth(nn.Module):  # type: ignore[misc]
    """Row-wise stochastic depth. Acts as identity in eval mode."""

    def __init__(self, drop_rate: float, scale_by_keep: bool = False) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        batch_size = x.shape[0]
        r = torch.rand((batch_size, 1, 1), device=x.device)
        keep_prob = 1.0 - self.drop_rate
        binary = torch.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary


class SkipInitChannelwise(nn.Module):  # type: ignore[misc]
    """LayerScale-style learned per-channel skip multiplier."""

    def __init__(self, channels: int, init_val: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.init_val = init_val
        self.skip = nn.Parameter(torch.ones(channels) * init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.skip


class PosEmbedding(nn.Module):  # type: ignore[misc]
    """Sine or learned positional embedding. Sine is parameter-free."""

    def __init__(self, d_model: int, max_len: int, use_sine: bool, patch_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_sine = use_sine
        self.patch_size = patch_size
        if not self.use_sine:
            self.embedding = nn.Embedding(max_len, d_model)
            self.register_buffer("position_ids", torch.arange(max_len))

    def forward(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        if self.use_sine:
            position_embeddings = sinusoidal_position_embedding(
                width // self.patch_size,
                height // self.patch_size,
                self.d_model,
                x.dtype,
                x.device,
            )
        else:
            position_embeddings = self.embedding(self.position_ids)
        return x + position_embeddings


class MLPBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, d_model: int, d_ff: int, stochdepth_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.stochdepth = (
            StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        if self.stochdepth is not None:
            x = self.stochdepth(x)
        return self.linear2(x)


class ViTBlock(nn.Module):  # type: ignore[misc]
    """Pre-norm ViT block with fused QKV, scaled dot-product attention,
    LayerScale, and optional stochastic depth.

    Parameter names (``norm1``, ``qkv_proj``, ``out_proj``, ``skip_init1``,
    ``norm2``, ``mlp.linear1``, ``mlp.linear2``, ``skip_init2``) match upstream
    so the safetensors state dict loads cleanly.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        layerscale_init: float,
        stochdepth_rate: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.skip_init1 = SkipInitChannelwise(channels=d_model, init_val=layerscale_init)
        self.stochdepth1 = (
            StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0 else None
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model, d_ff, stochdepth_rate)
        self.skip_init2 = SkipInitChannelwise(channels=d_model, init_val=layerscale_init)
        self.stochdepth2 = (
            StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, src_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads

        out = self.norm1(x)
        qkv = self.qkv_proj(out).split(self.d_model, dim=-1)
        q = qkv[0].view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)
        k = qkv[1].view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)
        v = qkv[2].view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, src_len, embed_dim)
        out = self.out_proj(out)
        out = self.skip_init1(out)
        if self.stochdepth1 is not None:
            out = self.stochdepth1(out)
        x = out + x

        out = self.norm2(x)
        out = self.mlp(out)
        out = self.skip_init2(out)
        if self.stochdepth2 is not None:
            out = self.stochdepth2(out)
        return out + x


class CNNLayerNorm(nn.Module):  # type: ignore[misc]
    """Channels-first ``LayerNorm`` for 4-D image tensors."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.norm(x)
        return x.transpose(1, 3)


class CNNStem(nn.Module):  # type: ignore[misc]
    """Mini DSL parser for upstream's ``cnn_stem`` config string.

    The string is ``;``-separated layers; each layer is ``<type>:<options>``
    where options is a comma-separated ``key=value`` list. Supported types:
    ``conv`` (with c/k/s/p), ``bn``, ``ln``, ``relu``, ``gelu``. The fancyfeast
    config uses ``conv:c=64;ln;relu;...;conv:c=768,s=1,k=1,p=0``.
    """

    def __init__(self, config: str) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        channels = 3
        for raw in config.split(";"):
            ty, _, opt_str = raw.partition(":")
            options = (
                {k: v for k, v in (o.split("=") for o in opt_str.split(","))} if opt_str else {}
            )
            if ty == "conv":
                out_c = int(options["c"])
                layers.append(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=out_c,
                        kernel_size=int(options.get("k", 3)),
                        stride=int(options.get("s", 2)),
                        padding=int(options.get("p", 1)),
                        bias=True,
                    )
                )
                channels = out_c
            elif ty == "bn":
                layers.append(nn.BatchNorm2d(channels))
            elif ty == "ln":
                layers.append(CNNLayerNorm(channels))
            elif ty == "relu":
                layers.append(nn.ReLU())
            elif ty == "gelu":
                layers.append(nn.GELU())
            else:
                msg = f"unknown CNN stem layer type: {ty!r}"
                raise ValueError(msg)
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _cait_layerscale_init(network_depth: int) -> float:
    """LayerScale init magnitude from the CaiT paper, by depth."""
    if network_depth <= 18:
        return 1e-1
    if network_depth <= 24:
        return 1e-5
    return 1e-6


class JoyTagViT(nn.Module):  # type: ignore[misc]
    """Inference-only port of ``Models.ViT`` from fancyfeast/joytag.

    The ``forward`` signature is simplified to a plain image tensor (instead
    of upstream's ``{'image': ...}`` batch dict) — call sites do
    ``model(tensor)`` and get logits of shape ``(B, n_tags)`` back. Sigmoid
    happens in ``JoyTagger.tag_image``.
    """

    def __init__(
        self,
        n_tags: int,
        image_size: int,
        num_blocks: int,
        patch_size: int,
        d_model: int,
        mlp_dim: int,
        num_heads: int,
        stochdepth_rate: float,
        use_sine: bool,
        layerscale_init: float | None = None,
        head_mean_after: bool = False,
        cnn_stem: str | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)

        self.n_tags = n_tags
        self.patch_size = patch_size
        self.head_mean_after = head_mean_after

        ls_init = _cait_layerscale_init(num_blocks) if layerscale_init is None else layerscale_init

        if cnn_stem is None:
            self.patch_embeddings: nn.Module = nn.Conv2d(
                in_channels=3,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
        else:
            self.patch_embeddings = CNNStem(cnn_stem)

        max_len = (image_size // patch_size) ** 2
        self.pos_embedding = PosEmbedding(d_model, max_len, use_sine=use_sine, patch_size=patch_size)

        self.blocks = nn.ModuleList(
            [ViTBlock(num_heads, d_model, mlp_dim, ls_init, stochdepth_rate) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_tags)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        if h % self.patch_size or w % self.patch_size:
            msg = (
                f"input H/W ({h}x{w}) must be divisible by patch_size ({self.patch_size})"
            )
            raise ValueError(msg)

        x = self.patch_embeddings(image)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_embedding(x, w, h)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if self.head_mean_after:
            x = self.head(x)
            return x.mean(dim=1)
        x = x.mean(dim=1)
        return self.head(x)


def build_joytag_vit(config: dict[str, Any]) -> JoyTagViT:
    """Construct a ``JoyTagViT`` from upstream's ``config.json`` shape.

    Drops the ``class`` and ``loss_type`` keys (training-only) before feeding
    the rest as kwargs. Unknown keys are filtered to keep this forward-compat
    with upstream config additions.
    """
    accepted = {
        "n_tags",
        "image_size",
        "num_blocks",
        "patch_size",
        "d_model",
        "mlp_dim",
        "num_heads",
        "stochdepth_rate",
        "use_sine",
        "layerscale_init",
        "head_mean_after",
        "cnn_stem",
    }
    kwargs = {k: v for k, v in config.items() if k in accepted}
    return JoyTagViT(**kwargs)


def load_joytag_state_dict(model: JoyTagViT, state_dict: dict[str, torch.Tensor]) -> None:
    """Load weights into ``model`` with the same legacy-tag stripping upstream does.

    Upstream supports old checkpoints that had 9 extra rating/score outputs
    appended to the head. Trim them so newer ``n_tags``-only models load.
    """
    head_w = state_dict.get("head.weight")
    if head_w is not None and head_w.shape[0] == model.n_tags + 9:
        state_dict["head.weight"] = head_w[: model.n_tags]
        bias = state_dict.get("head.bias")
        if bias is not None:
            state_dict["head.bias"] = bias[: model.n_tags]
    model.load_state_dict(state_dict, strict=True)


__all__ = [
    "JoyTagViT",
    "build_joytag_vit",
    "load_joytag_state_dict",
]
