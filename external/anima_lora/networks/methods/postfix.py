# Postfix tuning network module for Anima LLM Adapter.
#
# Learns N continuous vectors spliced into the cached adapter output
# (T5-compatible space). These discover quality signals in embedding space
# that improve generation across all artist tags.
#
# Two modes:
#   "postfix" — free-parameter postfix. A single K×D tensor shared across
#               the batch, appended (or scattered) into the cached adapter
#               output. Compatible with cache_llm_adapter_outputs.
#   "cond"    — caption-conditional postfix with structural orthogonality.
#               Always uses an orthonormal basis (random or SVD-of-cached-TE)
#               and a Cayley-rotated rank-K subspace; cond_mlp reads the
#               pooled caption and emits (S(c), λ(c)) per caption so
#               `postfix(c) = Cayley(S(c) − S(c).T) @ basis · λ(c)` and
#               `postfix(c) @ postfix(c).T = λ(c)² · I_K` structurally.

import glob
import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Default Qwen3 hidden dimension
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
    num_postfix_tokens = network_dim if network_dim is not None else 8

    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    mode = kwargs.get("mode", "postfix")
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    cond_hidden_dim = int(kwargs.get("cond_hidden_dim", 256))
    ortho_basis = str(kwargs.get("ortho_basis", "random"))
    te_cache_dir = kwargs.get("te_cache_dir", None)
    svd_num_files = int(kwargs.get("svd_num_files", 256))
    ortho_basis_seed = int(kwargs.get("ortho_basis_seed", 0))
    lambda_init = float(kwargs.get("lambda_init", 0.0))

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        splice_position=splice_position,
        cond_hidden_dim=cond_hidden_dim,
        ortho_basis=ortho_basis,
        te_cache_dir=te_cache_dir,
        svd_num_files=svd_num_files,
        ortho_basis_seed=ortho_basis_seed,
        lambda_init=lambda_init,
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
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    metadata_mode = None
    metadata_splice = None
    metadata_cond_hidden = None
    metadata_ortho_basis = None
    metadata_lambda_init = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_mode = meta.get("ss_mode")
            metadata_splice = meta.get("ss_splice_position")
            metadata_cond_hidden = meta.get("ss_cond_hidden_dim")
            metadata_ortho_basis = meta.get("ss_ortho_basis")
            metadata_lambda_init = meta.get("ss_lambda_init")

    has_cond = any(k.startswith("cond_mlp.") for k in weights_sd.keys())
    has_postfix_embeds = "postfix_embeds" in weights_sd
    has_ortho_basis = "ortho_basis" in weights_sd

    # Reject every mode we used to support but no longer do, with named errors.
    if "prefix_embeds" in weights_sd or metadata_mode == "prefix":
        raise ValueError(
            "Prefix-mode checkpoints are no longer supported. Train a fresh "
            "postfix or cond+ortho checkpoint."
        )
    if metadata_mode == "cond-timestep" or any(
        k.startswith("sigma_mlp.") for k in weights_sd.keys()
    ):
        raise ValueError(
            "cond-timestep checkpoints are no longer supported. Retrain in "
            "mode='cond' (which always uses ortho)."
        )
    if "ortho_S" in weights_sd or "ortho_lambda_global" in weights_sd:
        raise ValueError(
            "Standalone postfix+ortho checkpoints (ortho_S + ortho_lambda_global) "
            "are no longer supported. Retrain in mode='cond'."
        )

    if has_cond:
        mode = "cond"
    elif has_postfix_embeds:
        mode = "postfix"
    else:
        mode = metadata_mode or "postfix"

    if mode == "cond":
        # cond+ortho: LayerNorm(0) → Linear(1) → GELU(2) → Linear(3)
        # First Linear at cond_mlp.1; last Linear at cond_mlp.3 outputs
        # K(K-1)/2 + 1 (K comes from ortho_basis.shape[0]).
        w0 = weights_sd.get("cond_mlp.1.weight")
        w2 = weights_sd.get("cond_mlp.3.weight")
        if w0 is None or w2 is None:
            raise ValueError(
                "cond mode requires cond_mlp.1.weight and cond_mlp.3.weight "
                f"(got keys: {[k for k in weights_sd.keys() if 'cond_mlp' in k]})"
            )
        cond_hidden_dim = w0.shape[0]
        embed_dim = w0.shape[1]
        if not has_ortho_basis:
            raise ValueError(
                "cond mode requires 'ortho_basis' (legacy non-ortho cond "
                "checkpoints are no longer loadable — retrain in mode='cond')"
            )
        basis = weights_sd["ortho_basis"]
        num_postfix_tokens, basis_D = basis.shape
        if basis_D != embed_dim:
            raise ValueError(
                f"cond basis dim {basis_D} != cond_mlp embed_dim {embed_dim}"
            )
        expected_n_out = num_postfix_tokens * (num_postfix_tokens - 1) // 2 + 1
        if w2.shape[0] != expected_n_out:
            raise ValueError(
                f"cond cond_mlp last-layer dim {w2.shape[0]} != expected "
                f"{expected_n_out} for K={num_postfix_tokens}"
            )
    elif mode == "postfix":
        postfix_weight = weights_sd.get("postfix_embeds")
        if postfix_weight is None:
            raise ValueError(
                f"postfix mode requires 'postfix_embeds' (got keys: "
                f"{list(weights_sd.keys())[:10]})"
            )
        num_postfix_tokens, embed_dim = postfix_weight.shape
        cond_hidden_dim = int(metadata_cond_hidden) if metadata_cond_hidden else 256
    else:
        raise ValueError(
            f"Unknown postfix mode {mode!r} — expected 'postfix' or 'cond'."
        )

    splice_position = metadata_splice or "end_of_sequence"

    # ortho-side load-time kwargs: te_cache_dir intentionally defaults to None
    # so the __init__ path uses a throwaway random basis (load_weights
    # immediately overwrites it from the on-disk fp32 buffer).
    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        splice_position=splice_position,
        cond_hidden_dim=cond_hidden_dim,
        ortho_basis=metadata_ortho_basis or "random",
        te_cache_dir=kwargs.get("te_cache_dir", None),
        svd_num_files=int(kwargs.get("svd_num_files", 256)),
        ortho_basis_seed=int(kwargs.get("ortho_basis_seed", 0)),
        lambda_init=float(metadata_lambda_init) if metadata_lambda_init else 0.0,
    )
    return network, weights_sd


def _build_svd_te_basis(
    cache_dir: str,
    K: int,
    D: int,
    num_files: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Top-K right singular vectors of a sample of cached adapter outputs,
    row-shuffled deterministically.

    Reads `*_anima_te.safetensors` under ``cache_dir``, masks padding via
    `attn_mask_v0`, accumulates non-padding rows into an (M, D) matrix, runs
    full SVD, and returns the top-K rows of V_h (the K right singular vectors
    with the largest singular values). The K rows are row-orthonormal (V_h has
    orthonormal rows by construction).

    Row-shuffle (deterministic from `seed`) breaks the "slot-0 is the principal
    direction" inductive bias that would otherwise let the optimizer collapse
    its budget onto the top slot — same spirit as OrthoHydra's `e mod B`
    interleaving (`networks/lora_modules/hydra.py:95`), where each band
    receives a representative spread of singular slices instead of binding
    band 0 to the top slice.
    """
    if K > D:
        raise ValueError(
            f"cond mode requires K ({K}) ≤ D ({D}); cannot build K orthonormal "
            "rows in a D-dim space"
        )

    from safetensors.torch import load_file as _load_file

    files = sorted(
        glob.glob(
            os.path.join(cache_dir, "**", "*_anima_te.safetensors"),
            recursive=True,
        )
    )
    if not files:
        raise FileNotFoundError(
            f"ortho_basis='svd_te' requires cached *_anima_te.safetensors files "
            f"under {cache_dir!r} (run `make preprocess-te` first)"
        )

    rng = torch.Generator().manual_seed(int(seed))
    if len(files) > num_files:
        idx = torch.randperm(len(files), generator=rng)[:num_files].tolist()
        files = [files[i] for i in sorted(idx)]

    chunks: list[torch.Tensor] = []
    for path in files:
        sd = _load_file(path)
        emb = sd["crossattn_emb_v0"].float()  # (S, D)
        mask = sd["attn_mask_v0"].bool()       # (S,)
        if emb.shape[-1] != D:
            raise ValueError(
                f"cached embed dim {emb.shape[-1]} != requested D={D} (file: {path})"
            )
        if mask.any():
            chunks.append(emb[mask])

    if not chunks:
        raise RuntimeError(f"no non-padding tokens found across {len(files)} cached files")

    A = torch.cat(chunks, dim=0)  # (M, D)
    # full_matrices=False → V_h: (min(M, D), D); top-K rows are the K right
    # singular vectors with the largest singular values.
    _U, _S, V_h = torch.linalg.svd(A, full_matrices=False)
    if V_h.shape[0] < K:
        raise RuntimeError(
            f"svd_te: only {V_h.shape[0]} singular vectors available (< K={K}); "
            "use more cached files or smaller K"
        )
    top = V_h[:K].contiguous()  # (K, D), row-orthonormal

    # Deterministic row-shuffle: scrambles the "slot k = k-th principal
    # direction" ordering so the optimizer can't latch onto slot 0.
    perm = torch.randperm(K, generator=rng)
    return top[perm].contiguous()


def _make_orthonormal_basis(
    K: int,
    D: int,
    kind: str = "random",
    *,
    te_cache_dir: Optional[str] = None,
    svd_num_files: int = 256,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build a (K, D) row-orthonormal basis (K rows, D-dim each).

    QR on a (D, K) Gaussian matrix gives Q with orthonormal columns; transpose
    to get K row-orthonormal vectors in R^D. Requires K ≤ D.

    Supports two basis kinds:
      - ``"random"``: QR of a Gaussian (D, K) matrix.
      - ``"svd_te"``: top-K right singular vectors of cached
        ``_anima_te.safetensors`` adapter outputs under ``te_cache_dir``,
        row-shuffled with ``seed``. See ``_build_svd_te_basis``.
    """
    if K > D:
        raise ValueError(
            f"cond mode requires K ({K}) ≤ D ({D}); cannot build K orthonormal "
            "rows in a D-dim space"
        )
    if kind == "random":
        M = torch.randn(D, K, generator=generator)
        Q, _R = torch.linalg.qr(M)  # Q: (D, K), columns orthonormal
        return Q.T.contiguous()  # (K, D), rows orthonormal
    if kind == "svd_te":
        if te_cache_dir is None:
            raise ValueError(
                "ortho_basis='svd_te' requires te_cache_dir kwarg (path to a directory "
                "of cached *_anima_te.safetensors files, typically post_image_dataset/lora)"
            )
        return _build_svd_te_basis(
            te_cache_dir, K, D, num_files=svd_num_files, seed=seed
        )
    raise NotImplementedError(
        f"ortho_basis={kind!r}: only 'random' and 'svd_te' are implemented. "
        "See docs/proposal/orthogonal_postfix.md §Basis choice."
    )


class PostfixNetwork(AdapterNetworkBase):
    network_module = "networks.methods.postfix"
    network_spec = "postfix"

    def __init__(
        self,
        num_postfix_tokens: int,
        embed_dim: int,
        multiplier: float = 1.0,
        mode: str = "postfix",
        splice_position: str = "end_of_sequence",
        cond_hidden_dim: int = 256,
        ortho_basis: str = "random",
        te_cache_dir: Optional[str] = None,
        svd_num_files: int = 256,
        ortho_basis_seed: int = 0,
        lambda_init: float = 0.0,
    ):
        super().__init__()
        if mode not in ("postfix", "cond"):
            raise ValueError(
                f"mode must be 'postfix' or 'cond', got {mode!r}"
            )
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', got {splice_position!r}"
            )

        self.num_postfix_tokens = num_postfix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier
        self.mode = mode
        self.splice_position = splice_position
        self.cond_hidden_dim = cond_hidden_dim
        self.ortho_basis_kind = str(ortho_basis)
        self.te_cache_dir = te_cache_dir
        self.svd_num_files = int(svd_num_files)
        self.ortho_basis_seed = int(ortho_basis_seed)
        self.lambda_init = float(lambda_init)

        # Init scale matches the T5-compatible adapter output space (post-RMSNorm, std ≈ 1.0).
        init_std = 1.0

        if mode == "cond":
            # Caption-conditional + structurally orthogonal:
            #   cond_mlp: LN(D_pooled) → hidden → K(K-1)/2 + 1 scalars per caption
            #     - first K(K-1)/2 outputs → strict upper-tri of S(c) ∈ R^{K×K}
            #     - last 1 output → λ(c) (per-caption magnitude)
            #   postfix(c) = Cayley(S(c) − S(c).T) @ basis · λ(c)   (K, D)
            # Structurally `postfix(c) @ postfix(c).T = λ(c)² · I_K` per caption.
            #
            # Pre-norm on the pooled input: mean-pooled T5 outputs sit on a
            # narrow cone (cos μ ≈ 0.84 across captions, dominated by a corpus
            # DC offset). Default-init Linear would project that DC across every
            # hidden unit, swamping caption deltas — bench 20260511-1004 showed
            # cond_mlp[0] mapping cos 0.84 → 0.997 in a single step, the worst
            # single jump in the network. LayerNorm strips the DC + uniformizes
            # the input scale before the first Linear sees it; γ=1, β=0 init
            # keeps the rest of the cond_mlp's zero-init behavior intact (final
            # Linear still starts at zero → empty postfix at step 0).
            basis_kind_for_init = self.ortho_basis_kind
            if basis_kind_for_init == "svd_te" and self.te_cache_dir is None:
                logger.info(
                    "ortho_basis='svd_te' but te_cache_dir is None — using a "
                    "random throwaway basis at __init__ (will be overwritten "
                    "by load_weights). Pass te_cache_dir at training time to "
                    "actually compute the SVD basis."
                )
                basis_kind_for_init = "random"
            basis = _make_orthonormal_basis(
                num_postfix_tokens,
                embed_dim,
                kind=basis_kind_for_init,
                te_cache_dir=self.te_cache_dir,
                svd_num_files=self.svd_num_files,
                seed=self.ortho_basis_seed,
            )
            self.register_buffer("postfix_basis", basis)  # (K, D)

            n_skew = num_postfix_tokens * (num_postfix_tokens - 1) // 2
            n_out = n_skew + 1  # K(K-1)/2 rotation seed entries + 1 magnitude scalar
            self.cond_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, cond_hidden_dim),
                nn.GELU(),
                nn.Linear(cond_hidden_dim, n_out),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)
            # Non-zero λ_init: bias the λ(c) output channel so the postfix has
            # non-trivial magnitude at step 0. v2_ln (lambda_init=0) saw λ(c)
            # collapse from mean 0.50 (epoch 1) to mean 0.034 (epoch 2 final);
            # bench/postfix_ortho/results/20260511-1622-cond-v2-ln-final/. The
            # zero-init never gave the network an amplitude to *defend* — only
            # an amplitude to *grow from zero* — and any L2 / weight-decay
            # pressure on cond_mlp shrinks it back. Biasing the λ bias makes
            # the network choose between "keep using the postfix" and "kill
            # it" rather than "find a reason to start it." The skew-seed bias
            # entries stay zero (S(c)=0 → R=I → postfix(c) = basis · λ_init).
            if self.lambda_init != 0.0:
                with torch.no_grad():
                    self.cond_mlp[-1].bias[-1] = self.lambda_init
            # Strict-upper-tri index pairs + identity matrix for the S(c)
            # reconstruction in append_postfix. Registered as persistent=False
            # buffers so they live on the module's device after .to(...) (no
            # per-forward .to() round-trip) and don't show up in state_dict.
            triu = torch.triu_indices(num_postfix_tokens, num_postfix_tokens, offset=1)
            self.register_buffer("_S_triu_i", triu[0].contiguous(), persistent=False)
            self.register_buffer("_S_triu_j", triu[1].contiguous(), persistent=False)
            self.register_buffer(
                "_eye_K",
                torch.eye(num_postfix_tokens, dtype=torch.float32),
                persistent=False,
            )

            total_params = sum(p.numel() for p in self.cond_mlp.parameters())
            logger.info(
                f"PostfixNetwork: cond+ortho({self.ortho_basis_kind}) mode — "
                f"K={num_postfix_tokens} structurally-orthogonal slots × dim {embed_dim}, "
                f"hidden {cond_hidden_dim}, splice={self.splice_position}, pre-norm on pooled input, "
                f"lambda_init={self.lambda_init}, "
                f"{total_params} params (cond_mlp last layer outputs "
                f"{n_skew} skew-seed + 1 lambda(c) = {n_out}; basis frozen)"
            )
        else:
            # Plain postfix: T5-compatible postfix (appended to cached adapter output).
            self.postfix_embeds = nn.Parameter(
                torch.randn(num_postfix_tokens, embed_dim) * init_std
            )
            logger.info(
                f"PostfixNetwork: postfix mode — {num_postfix_tokens} tokens in T5-compatible space, "
                f"dim {embed_dim}, init_std={init_std}, splice={self.splice_position}, "
                f"{self.postfix_embeds.numel()} params"
            )

        self._last_postfix: Optional[torch.Tensor] = None
        self._last_cond_out: Optional[torch.Tensor] = None

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        # No monkey-patching needed — training loop handles postfix on cached crossattn_emb.
        logger.info(
            f"{self.mode} mode: {self.num_postfix_tokens} learned tokens will be appended to "
            f"cached adapter output (T5-compatible space)"
        )

    def _apply(self, fn, recurse=True):
        """Preserve fp32 dtype on fp32-required buffers across .to()/.bfloat16().

        Both `postfix_basis` and `_eye_K` feed the Cayley solve, which needs
        fp32 for the orthogonality gate (‖postfix @ postfix.T − λ²·I‖_F < 1e-4
        in the proposal). save_weights also pins the basis at fp32. Without
        this override, `network.to(torch.bfloat16)` (used under `full_bf16`)
        would silently downcast both and break the property.

        Device moves still pass through — the cast back to fp32 below preserves
        whichever device `fn` placed the buffer on.
        """
        out = super()._apply(fn, recurse=recurse)
        for name in ("postfix_basis", "_eye_K"):
            buf = self._buffers.get(name)
            if buf is not None and buf.dtype != torch.float32:
                self._buffers[name] = buf.to(torch.float32)
        return out

    def _compute_ortho_cond_postfix(
        self, pooled: torch.Tensor, target_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure cond+ortho path: pooled (B, D) → (cond_out (B, n_out), postfix (B, K, D)).

        Designed as a `torch.compile` target — no state mutation (caller writes
        `_last_*` after the call), no Python branching on tensor values, all
        buffers are read-only. Shapes are static once K, embed_dim, and B are
        fixed by bucketing, so the compile boundary is shape-static.

        cond_mlp runs in the autocast dtype (bf16 under training); Cayley solve
        + matmul run in fp32 against the dtype-pinned `_eye_K` and
        `postfix_basis` buffers (see `_apply` override).
        """
        K = self.num_postfix_tokens
        B = pooled.shape[0]
        cond_out = self.cond_mlp(pooled)  # (B, K(K-1)/2 + 1)
        n_skew = K * (K - 1) // 2
        S_seed = cond_out[:, :n_skew].float()
        lam_c = cond_out[:, -1].float()

        S_c = pooled.new_zeros(B, K, K, dtype=torch.float32)
        S_c[:, self._S_triu_i, self._S_triu_j] = S_seed
        A = S_c - S_c.transpose(-1, -2)
        R = torch.linalg.solve(self._eye_K + A, self._eye_K - A)  # (B, K, K)
        rotated = torch.matmul(R, self.postfix_basis)  # (B, K, D); both fp32
        postfix = (rotated * lam_c[:, None, None]).to(target_dtype)
        return cond_out, postfix

    def compile_hot_path(
        self, backend: str = "inductor", mode: Optional[str] = None
    ) -> None:
        """torch.compile the cond+ortho hot path inside `append_postfix`.

        Targets `_compute_ortho_cond_postfix`, which is shape-static once K,
        embed_dim, and B are fixed by bucketing (`dynamic=False` is safe — same
        justification as `AnimaDiT.compile_core`). Fuses the cond_mlp + Cayley
        + matmul + cast sequence (~15 small kernels → 1 graph at B=1), removing
        the per-step launch overhead from this eager-Python region.

        No-op outside cond mode — plain postfix is a single broadcast.
        """
        if self.mode != "cond":
            return
        compile_kwargs: dict = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        self._compute_ortho_cond_postfix = torch.compile(  # type: ignore[method-assign]
            self._compute_ortho_cond_postfix, **compile_kwargs
        )
        logger.info(
            f"PostfixNetwork: compiled cond+ortho hot path "
            f"(backend={backend}, mode={mode})"
        )

    def append_postfix(
        self,
        crossattn_emb: torch.Tensor,
        crossattn_seqlens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Splice learned postfix vectors into crossattn_emb (overwrites zero-padding slots).

        Splice position controlled by self.splice_position:
          - "end_of_sequence": place at [S-K, S). Caption-position-agnostic; preserves the
            strongest front-of-padding sinks intact.
          - "front_of_padding": place at [seqlens[i], seqlens[i]+K). Caption-position-aware;
            displaces the strongest sinks. Legacy behavior.

        In "cond" mode the postfix vectors are computed per-sample by pooling content
        slots through cond_mlp (maxabs-pool over content tokens). In "postfix" mode
        they come from a single learned parameter tensor shared across the batch.

        Args:
            crossattn_emb: [B, S, D] cached adapter output (zero-padded after real tokens)
            crossattn_seqlens: [B] number of real text tokens per batch element
        """
        K = self.num_postfix_tokens
        B, S, D = crossattn_emb.shape

        if self.mode == "cond":
            pos = torch.arange(S, device=crossattn_emb.device).unsqueeze(0)  # [1, S]
            content_mask = pos < crossattn_seqlens.unsqueeze(1)  # [B, S] bool

            # Maxabs-pool over content slots: pick per channel the token with
            # the largest |·| (sign preserved). Diagnostic bench 20260511-1004
            # showed mean-pool produces cos μ=0.84 across captions (vs 0.22
            # for maxabs) — T5 outputs have always-positive "baseline"
            # channels that mean/max averaging drags every caption onto;
            # caption-distinct signal lives in both positive AND negative
            # deflections, which maxabs preserves by picking by magnitude.
            # Padding is zero, so we set its |·| to -1 so it can never win
            # the argmax against any non-zero content token. In-place fill
            # on the abs() result avoids a second [B,S,D] allocation.
            abs_emb = crossattn_emb.abs()
            abs_emb.masked_fill_(~content_mask.unsqueeze(-1), -1.0)
            idx = abs_emb.argmax(dim=1, keepdim=True)  # [B, 1, D]
            pooled = crossattn_emb.gather(dim=1, index=idx).squeeze(1)  # [B, D]

            # cond_mlp predicts (S(c), λ(c)) per caption. Body lives in
            # `_compute_ortho_cond_postfix` so it's a clean `torch.compile`
            # target (see `compile_hot_path`). State writes for diagnostics
            # stay here, outside the compiled region.
            cond_out, postfix = self._compute_ortho_cond_postfix(
                pooled, crossattn_emb.dtype
            )
            self._last_cond_out = cond_out
            self._last_postfix = postfix
        else:
            postfix = (
                self.postfix_embeds.unsqueeze(0)
                .expand(B, -1, -1)
                .to(dtype=crossattn_emb.dtype, device=crossattn_emb.device)
            )

        if self.splice_position == "end_of_sequence":
            # Overwrite the last K slots of the zero-padding region with the postfix.
            # torch.cat preserves autograd on both sides.
            return torch.cat([crossattn_emb[:, : S - K, :], postfix], dim=1)

        # front_of_padding: place K postfix tokens at [seqlens[i], seqlens[i]+K) per sample
        offsets = crossattn_seqlens.long().unsqueeze(1) + torch.arange(
            K, device=crossattn_emb.device
        )  # [B, K]
        idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        return crossattn_emb.scatter(1, idx, postfix)

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references between training/validation steps.

        Under ``compile_inductor_mode="reduce-overhead"`` (cudagraph_trees),
        ``_last_postfix`` / ``_last_cond_out`` hold tensors produced inside
        ``_compute_ortho_cond_postfix`` (compiled hot path) — those live in
        the cudagraph memory pool. Keeping references across the step
        boundary pins the pool, forcing re-records or silent eager fallback.
        Caller invokes this right before
        ``torch.compiler.cudagraph_mark_step_begin()`` (see ``train.py`` and
        the validation loop) so the pool can recycle on the next iteration.

        Especially load-bearing at the train→eval→train boundary (first
        epoch's end-of-epoch validation), where stale train-side references
        would otherwise persist across the val pass and demote subsequent
        training steps to eager — observed as a one-time epoch 1 → epoch 2
        slowdown from ~510 ms/step to ~900 ms/step.
        """
        self._last_postfix = None
        self._last_cond_out = None

    def get_trainable_params(self):
        if self.mode == "cond":
            return list(self.cond_mlp.parameters())
        return [self.postfix_embeds]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        if self.mode == "cond":
            params = [{"params": list(self.cond_mlp.parameters()), "lr": lr}]
            descriptions = ["cond_mlp"]
        else:
            params = [{"params": [self.postfix_embeds], "lr": lr}]
            descriptions = ["postfix_embeds"]
        return params, descriptions

    def state_dict_for_save(self, dtype):
        if self.mode == "cond":
            sd = {
                f"cond_mlp.{k}": v.detach().clone().cpu().to(dtype)
                for k, v in self.cond_mlp.state_dict().items()
            }
            # Frozen SVD basis must be persisted at fp32 — bf16 truncation
            # blows the orthogonality gate (‖postfix @ postfix.T - λ²·I‖_F).
            sd["ortho_basis"] = self.postfix_basis.detach().clone().cpu().float()
            return sd
        return {
            "postfix_embeds": self.postfix_embeds.detach().clone().cpu().to(dtype)
        }

    def metadata_fields(self) -> dict[str, str]:
        meta: dict[str, str] = {
            "ss_num_postfix_tokens": str(self.num_postfix_tokens),
            "ss_embed_dim": str(self.embed_dim),
            "ss_mode": self.mode,
            "ss_splice_position": self.splice_position,
        }
        if self.mode == "cond":
            meta["ss_cond_hidden_dim"] = str(self.cond_hidden_dim)
            meta["ss_ortho_basis"] = self.ortho_basis_kind
            meta["ss_ortho_basis_seed"] = str(self.ortho_basis_seed)
            if self.te_cache_dir is not None:
                meta["ss_te_cache_dir"] = str(self.te_cache_dir)
            meta["ss_svd_num_files"] = str(self.svd_num_files)
            meta["ss_lambda_init"] = str(self.lambda_init)
        return meta

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        if self.mode == "cond":
            mlp_sd = {
                k[len("cond_mlp.") :]: v
                for k, v in weights_sd.items()
                if k.startswith("cond_mlp.")
            }
            if not mlp_sd:
                raise ValueError(
                    "No 'cond_mlp.*' keys found in weights file for cond mode"
                )
            missing, unexpected = self.cond_mlp.load_state_dict(mlp_sd, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"cond_mlp load_state_dict mismatch: missing={missing}, unexpected={unexpected}"
                )
            basis_w = weights_sd.get("ortho_basis")
            if basis_w is None:
                raise ValueError(
                    "cond mode requires 'ortho_basis' (got keys: "
                    f"{[k for k in weights_sd.keys() if k.startswith('ortho_')]})"
                )
            self.postfix_basis.copy_(basis_w.to(self.postfix_basis.dtype))
            logger.info(
                f"Loaded cond+ortho: K={self.num_postfix_tokens} D={self.embed_dim} "
                f"basis={self.ortho_basis_kind} (cond_mlp params: "
                f"{sum(p.numel() for p in self.cond_mlp.parameters())})"
            )
        else:
            weight = weights_sd.get("postfix_embeds")
            if weight is None:
                raise ValueError(
                    "No 'postfix_embeds' key found in weights file for postfix mode"
                )
            self.postfix_embeds.data.copy_(weight)
            logger.info(f"Loaded postfix weights: {self.postfix_embeds.shape}")
