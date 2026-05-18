# Per-layer time-indexed soft tokens — SoftREPA-style parameterization (without
# the contrastive loss). DiT is frozen; trains a small bank of K continuous
# vectors per (layer, t-bucket) pair, prepended (via end-of-sequence overwrite)
# to crossattn_emb at each block independently.
#
# Reference: Lee et al., "Aligning Text to Image in Diffusion Models is Easier
# Than You Think" (NeurIPS 2025) — arXiv:2503.08250. We adopt only their
# parameterization (per-layer × per-t soft tokens), trained under plain flow-
# matching loss; the contrastive InfoNCE objective is intentionally skipped.
# In Anima the DiT is cross-attention (not joint-stream MM-DiT like SD3), so
# crossattn_emb does not evolve through blocks — each block independently
# receives crossattn_emb extended with its own layer tokens. No strip/re-prepend
# dance is needed.
#
# Splice strategy: end-of-sequence overwrite of zero-padding (K tail slots),
# preserving static crossattn_emb shape so `_run_blocks` torch.compile stays
# happy. Zero-padded positions act as cross-attention sinks (see Anima's text-
# encoder padding invariant), so writing tokens into them gives them attention
# mass without changing seqlen.
#
# Why a separate module from postfix.py: postfix splices once at the cached
# adapter output (training-time and inference-time). Soft tokens splice per-
# block via monkey-patched Block.forward (ReFT-pattern), a fundamentally
# different surface.
#
# v1: training only. Inference would require the per-step splice to be re-run
# inside the denoising loop; until that's wired up, save_weights still emits a
# usable file but inference.py will refuse to load it.

import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Anima cached crossattn_emb dimension (Qwen3 hidden size, post LLM-adapter).
DEFAULT_EMBED_DIM = 1024


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
    num_tokens = network_dim if network_dim is not None else 4
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    n_layers = int(kwargs.get("n_layers", 10))
    n_t_buckets = int(kwargs.get("n_t_buckets", 100))
    init_std = float(kwargs.get("init_std", 0.02))
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    contrastive_weight = float(kwargs.get("contrastive_weight", 0.0))
    contrastive_k = int(kwargs.get("contrastive_k", 1))
    contrastive_tau = float(kwargs.get("contrastive_tau", 1.0))
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=init_std,
        splice_position=splice_position,
        contrastive_weight=contrastive_weight,
        contrastive_k=contrastive_k,
        contrastive_tau=contrastive_tau,
        multiplier=multiplier,
    )
    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if for_inference:
        raise NotImplementedError(
            "soft_tokens v1 is training-only. Inference plumbing (per-step "
            "block hooks inside the denoising loop) is not wired up yet."
        )
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    tokens = weights_sd.get("tokens")
    t_offsets = weights_sd.get("t_offsets.weight")
    if tokens is None or t_offsets is None:
        raise ValueError(
            f"soft_tokens weight file must contain 'tokens' and 't_offsets.weight' "
            f"(got keys: {list(weights_sd.keys())[:8]})"
        )
    n_layers, num_tokens, embed_dim = tokens.shape
    n_t_buckets = t_offsets.shape[0]
    # Splice position is a runtime knob, not learned — read from metadata if
    # present, otherwise default. CLI kwargs win for post-hoc overrides.
    metadata_splice = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_splice = meta.get("ss_splice_position")
    splice_position = kwargs.get(
        "splice_position", metadata_splice or "end_of_sequence"
    )
    contrastive_weight = float(kwargs.get("contrastive_weight", 0.0))
    contrastive_k = int(kwargs.get("contrastive_k", 1))
    contrastive_tau = float(kwargs.get("contrastive_tau", 1.0))
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=0.0,  # weights are loaded; init_std doesn't matter
        splice_position=splice_position,
        contrastive_weight=contrastive_weight,
        contrastive_k=contrastive_k,
        contrastive_tau=contrastive_tau,
        multiplier=multiplier,
    )
    return network, weights_sd


class SoftTokensNetwork(AdapterNetworkBase):
    """Per-layer time-indexed soft tokens.

    Parameters:
      - tokens: (n_layers, K, D) — base per-layer tokens, small-std init.
      - t_offsets: Embedding(n_t_buckets, n_layers * D) — per-(t_bucket, layer)
        broadcast offset (one D-vector applied to every token in the layer).
        Zero-init so step 0 reproduces the un-time-conditioned base tokens.

    Param count: n_layers·K·D + n_t_buckets·n_layers·D
    With defaults (n_layers=10, K=4, D=1024, n_t_buckets=100): 40k + 1.0M ≈ 1.05M.
    """

    network_module = "networks.methods.soft_tokens"
    network_spec = "soft_tokens"

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        n_layers: int = 10,
        n_t_buckets: int = 100,
        init_std: float = 0.02,
        splice_position: str = "end_of_sequence",
        contrastive_weight: float = 0.0,
        contrastive_k: int = 1,
        contrastive_tau: float = 1.0,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        # Upper-bound check against actual block count happens in apply_to().
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if n_t_buckets <= 0:
            raise ValueError(f"n_t_buckets must be positive, got {n_t_buckets}")
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', "
                f"got {splice_position!r}"
            )

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_t_buckets = n_t_buckets
        self.splice_position = splice_position
        # Paper's InfoNCE objective. ``contrastive_k`` is the number of
        # in-batch negatives per anchor (paper had N-1 = full batch; we use a
        # tunable subset since each negative costs one extra DiT forward).
        # ``contrastive_tau`` constrains the unbounded MSE-as-similarity per
        # paper §3.1 ("exponential function to constrain the logit values").
        self.contrastive_weight = contrastive_weight
        self.contrastive_k = max(int(contrastive_k), 0)
        self.contrastive_tau = float(contrastive_tau)
        # Latest contrastive value cached by SoftTokensMethodAdapter for
        # logging via metrics(); ``_last_*`` mirrors the postfix pattern.
        self._last_contrastive_value: Optional[float] = None
        self.multiplier = multiplier

        self.tokens = nn.Parameter(
            torch.randn(n_layers, num_tokens, embed_dim) * init_std
        )
        # Per-(bucket, layer) D-vector offset. Broadcast across the K-token axis
        # at lookup so the bucket only has to learn one D-vector per layer per
        # bucket (not K). Zero-init = identity perturbation at step 0.
        self.t_offsets = nn.Embedding(n_t_buckets, n_layers * embed_dim)
        nn.init.zeros_(self.t_offsets.weight)

        # Step-scoped state set by append_postfix() once per forward pass and
        # consumed by the per-block hooks installed in apply_to(). Kept as a
        # plain attribute (not a buffer) — recreated each step, no need to
        # persist or move with .to(). _step_seqlens is only populated for
        # front_of_padding splice; end_of_sequence ignores it.
        self._step_layer_tokens: Optional[torch.Tensor] = None  # (n_layers, B, K, D)
        self._step_seqlens: Optional[torch.Tensor] = None  # (B,) int

        # Reverse-bookkeeping for apply_to(): keep references so we could
        # un-monkey-patch later (currently unused but cheap to track).
        self._block_refs: list[nn.Module] = []
        self._original_forwards: list = []

        n_token_params = self.tokens.numel()
        n_offset_params = self.t_offsets.weight.numel()
        cs_note = (
            f", contrastive(λ={contrastive_weight}, k={contrastive_k}, "
            f"τ={contrastive_tau})"
            if contrastive_weight > 0.0
            else ""
        )
        logger.info(
            f"SoftTokensNetwork: {n_layers} layers × {num_tokens} tokens × dim {embed_dim}, "
            f"{n_t_buckets} t-buckets, splice={splice_position}{cs_note} → "
            f"{n_token_params + n_offset_params} params "
            f"({n_token_params} base + {n_offset_params} t-offset)"
        )

    # Sentinel attribute so train.py's ``hasattr(network, "append_postfix")``
    # branch picks us up: train.py will then call append_postfix(..., timesteps=...)
    # at the right point in the forward loop, which we use only to compute the
    # step-scoped per-layer tokens. The crossattn_emb passes through unchanged
    # — splicing happens inside the per-block hooks below.
    @property
    def num_postfix_tokens(self) -> int:
        return self.num_tokens

    def _bucketize(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map sigma in [0, 1] (Anima convention) to integer buckets [0, n_t_buckets).

        Outside-range values are clamped, so callers don't need to pre-clamp.
        """
        t = timesteps.detach().float().flatten()
        idx = torch.floor(t * self.n_t_buckets).long()
        return idx.clamp(min=0, max=self.n_t_buckets - 1)

    def append_postfix(
        self,
        crossattn_emb: torch.Tensor,
        crossattn_seqlens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-step layer tokens and cache them for the block hooks.

        Returns crossattn_emb unchanged — splice happens per-block in the hooks
        installed by ``apply_to()``. We just piggyback on train.py's existing
        per-step trainer hook to receive timesteps.
        """
        if timesteps is None:
            raise ValueError(
                "soft_tokens requires timesteps (per-step) — train.py passes "
                "this automatically; inference path is not yet wired up"
            )
        B = crossattn_emb.shape[0]
        bucket_idx = self._bucketize(timesteps)  # (B,)
        # (B, n_layers * D) → (B, n_layers, D) → (B, n_layers, 1, D)
        offsets = self.t_offsets(bucket_idx).view(B, self.n_layers, self.embed_dim)
        # (n_layers, K, D) → (1, n_layers, K, D), broadcast over batch.
        base = self.tokens.unsqueeze(0)
        # (B, n_layers, K, D); offset broadcasts across K.
        per_step = base + offsets.unsqueeze(2)
        # Transpose to (n_layers, B, K, D) for cheap per-layer indexing in the
        # block hook closure.
        self._step_layer_tokens = per_step.transpose(0, 1).contiguous()
        # front_of_padding needs per-sample seqlens at hook time; end_of_sequence
        # ignores them. Cache regardless so the hook doesn't have to know which
        # mode is active (the splice branch reads or skips).
        self._step_seqlens = (
            crossattn_seqlens.detach().to(torch.long)
            if crossattn_seqlens is not None
            else None
        )
        return crossattn_emb

    def _make_block_hook(self, layer_idx: int, org_forward):
        """Closure that splices layer_idx's tokens into crossattn_emb tail.

        Block.forward signature (from library/anima/models.py:1179):
          forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                  rope_cos_sin=None, adaln_lora_B_T_3D=None)
        """
        K = self.num_tokens
        splice_position = self.splice_position
        net = self  # capture network for state lookup

        def hook(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            *args,
            **kwargs,
        ):
            step_tokens = net._step_layer_tokens
            if step_tokens is not None:
                # (B, K, D) for this layer. Cast to crossattn dtype/device.
                layer_tok = step_tokens[layer_idx].to(
                    dtype=crossattn_emb.dtype, device=crossattn_emb.device
                )
                S = crossattn_emb.shape[1]
                if S < K:
                    raise RuntimeError(
                        f"crossattn_emb seqlen {S} < num_tokens {K}; cannot splice"
                    )
                if splice_position == "end_of_sequence":
                    # Overwrite the K tail (zero-padding) slots. torch.cat
                    # preserves autograd through both branches.
                    crossattn_emb = torch.cat(
                        [crossattn_emb[:, : S - K, :], layer_tok], dim=1
                    )
                else:  # front_of_padding
                    # Place K tokens at [seqlens[i], seqlens[i]+K) per sample —
                    # displaces the strongest sinks. scatter() preserves grad
                    # on the written values.
                    seqlens = net._step_seqlens
                    if seqlens is None:
                        raise RuntimeError(
                            "front_of_padding splice requires crossattn_seqlens; "
                            "trainer must pass it to append_postfix()"
                        )
                    offsets = seqlens.to(crossattn_emb.device).unsqueeze(
                        1
                    ) + torch.arange(K, device=crossattn_emb.device)  # (B, K)
                    D = crossattn_emb.shape[-1]
                    idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
                    crossattn_emb = crossattn_emb.scatter(1, idx, layer_tok)
            return org_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                *args,
                **kwargs,
            )

        return hook

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder=True,
        apply_unet=True,
    ):
        """Monkey-patch the first n_layers DiT blocks with the splice hook."""
        blocks = getattr(unet, "blocks", None)
        if blocks is None:
            raise RuntimeError("unet has no .blocks attribute — not an Anima DiT?")
        if len(blocks) < self.n_layers:
            raise RuntimeError(
                f"unet has {len(blocks)} blocks but n_layers={self.n_layers}"
            )
        self._block_refs = []
        self._original_forwards = []
        for k in range(self.n_layers):
            block = blocks[k]
            org_forward = block.forward
            block.forward = self._make_block_hook(k, org_forward)
            self._block_refs.append(block)
            self._original_forwards.append(org_forward)
        logger.info(
            f"soft_tokens: monkey-patched first {self.n_layers} of {len(blocks)} "
            f"DiT blocks (end-of-sequence splice, K={self.num_tokens})"
        )

    # ── Standard adapter API ────────────────────────────────────────────

    def get_trainable_params(self):
        return [self.tokens, self.t_offsets.weight]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [{"params": self.get_trainable_params(), "lr": lr}]
        descriptions = ["soft_tokens(tokens+t_offsets)"]
        return params, descriptions

    def state_dict_for_save(self, dtype):
        return {
            "tokens": self.tokens.detach().clone().cpu().to(dtype),
            "t_offsets.weight": self.t_offsets.weight.detach().clone().cpu().to(dtype),
        }

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_tokens": str(self.num_tokens),
            "ss_embed_dim": str(self.embed_dim),
            "ss_n_layers": str(self.n_layers),
            "ss_n_t_buckets": str(self.n_t_buckets),
            "ss_splice_position": self.splice_position,
            "ss_contrastive_weight": str(self.contrastive_weight),
            "ss_contrastive_k": str(self.contrastive_k),
            "ss_contrastive_tau": str(self.contrastive_tau),
        }

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        if "tokens" not in weights_sd or "t_offsets.weight" not in weights_sd:
            raise ValueError(
                f"Missing required keys in soft_tokens checkpoint "
                f"(got: {list(weights_sd.keys())[:8]})"
            )
        self.tokens.data.copy_(weights_sd["tokens"])
        self.t_offsets.weight.data.copy_(weights_sd["t_offsets.weight"])
        logger.info(
            f"Loaded soft_tokens weights: tokens={tuple(self.tokens.shape)}, "
            f"t_offsets={tuple(self.t_offsets.weight.shape)}"
        )

    def metrics(self, ctx) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.contrastive_weight > 0.0 and self._last_contrastive_value is not None:
            v = float(self._last_contrastive_value)
            out["reg/soft_tokens_contrastive"] = v
            out["reg/soft_tokens_contrastive_weighted"] = self.contrastive_weight * v
        return out


# ───────────────────────────────────────────────── trainer integration


class SoftTokensMethodAdapter:
    """Trainer adapter: paper §3.1 InfoNCE forwards on top of plain FM.

    For each anchor i, run k extra DiT forwards with the *same* (x_t, ε, t)
    but with text features rolled by j ∈ {1, …, k} along the batch axis —
    giving k mismatched (x_i, y_{(i+j) mod B}) pairs. Build the (1+k)-way
    InfoNCE softmax over the negative diffusion-loss logits (paper eq. 13–14)
    with a constant temperature τ. The matched FM loss is unchanged — this
    adds a contrastive *regularizer* to the existing per-sample FM, not a
    replacement (the paper had FID regress under the contrastive weight on
    SD3 and Anima's narrower caption distribution likely needs a softer
    blend).

    The soft tokens themselves don't depend on text content, so the cached
    ``_step_layer_tokens`` from the matched forward is reused across negative
    forwards (the per-block hook re-splices into the *rolled* crossattn).
    Per-sample seqlens, however, do roll with the text — for FOP splice we
    refresh ``_step_seqlens`` before each negative forward.

    No-op when ``network.contrastive_weight <= 0``, in validation, when
    ``crossattn_emb`` is None (uncached path), or when batch size < 2 (no
    in-batch negatives possible).
    """

    name = "soft_tokens"

    def __init__(self) -> None:
        self._last_contrastive_value: Optional[float] = None

    def on_network_built(self, ctx) -> None:
        # Surface the adapter back to the network so its ``metrics()`` can
        # mirror our last contrastive value (mirroring postfix's pattern).
        if ctx.network is not None and not hasattr(ctx.network, "_soft_tokens_adapter"):
            ctx.network._soft_tokens_adapter = self

    def on_step_start(self, ctx, batch, *, is_train: bool) -> None:
        pass

    def prime_for_forward(self, ctx, batch, latents, *, is_train: bool) -> None:
        pass

    def validation_baselines(self):
        return []

    def on_epoch_end(self, ctx) -> None:
        pass

    def metrics(self, ctx) -> dict:
        return {}

    def extra_forwards(self, ctx, primary) -> Optional[dict]:
        if not primary.is_train:
            return None
        network = ctx.network
        weight = float(getattr(network, "contrastive_weight", 0.0) or 0.0)
        k = int(getattr(network, "contrastive_k", 0) or 0)
        if weight <= 0.0 or k <= 0:
            return None
        if primary.crossattn_emb is None:
            # Uncached text path — the rolled-text trick needs the (B, S, D)
            # tensor pre-splice. Fail loudly so users notice the gate.
            raise RuntimeError(
                "soft_tokens contrastive requires cached crossattn (set "
                "cache_llm_adapter_outputs=true in the method config)"
            )

        anima = primary.anima_call
        noisy = primary.noisy_model_input  # 5D
        timesteps = primary.timesteps
        target = primary.noise - primary.latents  # [B, C, H, W]
        padding_mask = primary.padding_mask
        kw = primary.forward_kwargs

        # The block hook inside Block.forward consumes whatever the most
        # recent ``append_postfix`` call cached on the network. The hook
        # re-runs in this scope when we call anima(...) with rolled text, so
        # the same per-(layer, t) tokens splice into the rolled crossattn —
        # exactly the paper's "shared (x_t, t, ε), varying y" setup. The
        # primary forward already wrote crossattn_emb that has the spliced
        # tokens at the splice position; we need the PRE-splice tensor to
        # roll. Recover it by undoing the splice on the cached tokens, then
        # roll, then let the block hook re-splice.
        spliced = primary.crossattn_emb  # post-splice from primary call
        B = spliced.shape[0]
        if B < 2:
            return None  # batchsize 1: no in-batch negatives

        K = network.num_tokens
        S = spliced.shape[1]
        # Reverse the EOS splice so we have the original (zero-padded) text;
        # for FOP we just trust the original cached path holds (the splice
        # writes K tokens at variable per-sample offsets, scatter is not
        # trivially invertible — gate FOP+contrastive off for now).
        if network.splice_position != "end_of_sequence":
            raise RuntimeError(
                "soft_tokens contrastive currently requires "
                "splice_position='end_of_sequence' (FOP scatter is not "
                "trivially invertible). Switch the config or set "
                "contrastive_weight=0."
            )
        # The K tail slots were originally zero-padding; restore that for
        # rolling so the rolled negative isn't contaminated by the matched
        # anchor's spliced tokens.
        zero_tail = torch.zeros(
            B, K, spliced.shape[2], dtype=spliced.dtype, device=spliced.device
        )
        original_text = torch.cat([spliced[:, : S - K, :], zero_tail], dim=1)

        # Per-sample MSE for matched (already computed by the trainer): same
        # functional form as paper eq. 13's logit numerator. We recompute on
        # the primary model_pred to avoid re-running the matched forward.
        # primary.model_pred is 5D; squeeze to 4D for per-sample MSE.
        v_match = primary.model_pred.squeeze(2)  # [B, C, H, W]
        per_sample_mse_list = [((v_match - target) ** 2).mean(dim=(1, 2, 3))]

        # k extra forwards with rolled text. Cyclic shifts ensure each anchor
        # sees k distinct negatives (provided B > k). When B <= k we run only
        # B-1 shifts (still informative).
        k_eff = min(k, B - 1)
        for j in range(1, k_eff + 1):
            rolled_text = torch.roll(original_text, shifts=j, dims=0)
            # Re-call append_postfix to refresh cached state (timesteps and
            # — for FOP — seqlens). The cached tokens are the same since
            # timesteps haven't changed, but this is the contract.
            seqlens_kwarg = (
                kw.get("crossattn_seqlens") if isinstance(kw, dict) else None
            )
            if seqlens_kwarg is None:
                seqlens_kwarg = torch.full(
                    (B,), S - K, dtype=torch.int32, device=rolled_text.device
                )
            rolled_seqlens = torch.roll(seqlens_kwarg, shifts=j, dims=0)
            network.append_postfix(rolled_text, rolled_seqlens, timesteps=timesteps)
            v_neg = anima(
                noisy,
                timesteps,
                rolled_text,
                padding_mask=padding_mask,
                **kw,
            ).squeeze(2)  # [B, C, H, W]
            per_sample_mse_list.append(((v_neg - target) ** 2).mean(dim=(1, 2, 3)))

        # Restore the primary's cached state so any downstream code that
        # peeks at network._step_layer_tokens / _step_seqlens sees the
        # matched-anchor view (the loss-side reads target=ε−x and won't
        # trigger another splice, but be defensive).
        primary_seqlens = (
            kw.get("crossattn_seqlens") if isinstance(kw, dict) else None
        )
        if primary_seqlens is None:
            primary_seqlens = torch.full(
                (B,), S - K, dtype=torch.int32, device=spliced.device
            )
        network.append_postfix(original_text, primary_seqlens, timesteps=timesteps)

        # InfoNCE: -log( exp(-mse_match/τ) / Σ_j exp(-mse_j/τ) )
        # Equivalent to log_softmax over (1+k_eff) candidates at index 0.
        mse_stack = torch.stack(per_sample_mse_list, dim=0)  # [(1+k_eff), B]
        logits = -mse_stack / max(network.contrastive_tau, 1e-6)
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        contrastive_per_sample = -log_probs[0]  # [B]
        contrastive_loss = contrastive_per_sample.mean()

        # Cache scalar value for metrics().
        network._last_contrastive_value = float(contrastive_loss.detach().item())
        self._last_contrastive_value = network._last_contrastive_value

        return {"soft_tokens": {"contrastive_loss": contrastive_loss}}
