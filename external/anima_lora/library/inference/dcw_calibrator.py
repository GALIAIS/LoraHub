"""DCW online calibrator — produces a per-step λ for the post-step DCW correction.

Loads a fusion-head safetensors artifact (head weights + standardization stats),
observes the LL-band Haar norm of the post-CFG ``noise_pred`` over the first
``k_warmup`` steps, fires the MLP at step ``k_warmup`` to predict the per-prompt
LSQ-optimal scalar λ̂*_p, then applies::

    λ_i = baseline_lambda · (1 − σ_i)                                      [all i]
        + α_eff · gain · (1 − σ_i)        for target_start ≤ i < target_end

clamped to ±0.05. ``baseline_lambda`` defaults to 0 for legacy artifacts —
non-zero means the head was trained on data already corrected with that
scalar, so α̂ is a residual on top of it (kills the warmup dead zone since
the baseline applies on every step, including i < target_start).

Schema compat:

* ``dcw_v5_lambda_scalar`` (default post-cleanup) — head reads ``(c_pool,
  g_obs[:k], aux)``; ``g_obs`` = ``haar_LL_norm(noise_pred)`` per warmup step.
* ``dcw_v6_fei_replace`` — head's ``g_obs`` slot is fed 2-band FEI low-band
  energy on the pre-forward latent instead of Haar norms; trainer redirected
  ``g_obs_mean/std`` onto FEI stats so loading is bit-equivalent to v5 except
  for the input source. Caller must invoke ``record_latent_pre_forward(i, z_t)``
  before each warmup-step model forward.
* ``dcw_v6_fei_concat`` — head adds a parallel ``fei[:k]`` column alongside
  the original ``g_obs[:k]``; both are needed per warmup step.
* ``dcw_v4_fusion_head`` — legacy.

Pre-``lambda_scalar`` v4 artifacts (``target_kind=alpha_residual``) are rejected
at load time — they need either a retrain or a ``git checkout`` to the
pre-cleanup controller.

The calibrator is **inactive** (``is_active == False``) when the artifact
fails to load or ``setup`` hits an empty embed mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from library.runtime.fei import compute_fei_2band, fei_sigma_low
from networks.dcw import FusionHead, haar_LL_norm

logger = logging.getLogger(__name__)

_VALID_SCHEMAS = (
    "dcw_v5_lambda_scalar",
    "dcw_v6_fei_replace",
    "dcw_v6_fei_concat",
    "dcw_v4_fusion_head",
)
_LAMBDA_CLAMP = 0.05
# Matches scripts/dcw/trajectory.py default (data-collection-time divisor).
# Hardcoded because the trainer does not stamp this into artifact metadata;
# if the bench script's default ever changes, mirror the bump here.
_FEI_SIGMA_LOW_DIV = 4.0


class OnlineDCWCalibrator:
    def __init__(
        self,
        head: FusionHead,
        centroid: torch.Tensor,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
        g_obs_mean: torch.Tensor,
        g_obs_std: torch.Tensor,
        k_warmup: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        target_start: Optional[int] = None,
        target_end: Optional[int] = None,
        c_pool_norm: str = "none",
        c_pool_mean: Optional[torch.Tensor] = None,
        c_pool_std: Optional[torch.Tensor] = None,
        baseline_lambda: float = 0.0,
        fei_obs: str = "off",
        fei_k: int = 0,
        fei_mean: Optional[torch.Tensor] = None,
        fei_std: Optional[torch.Tensor] = None,
    ):
        self.head = head.to(device=device, dtype=dtype).eval()
        self.centroid = centroid.to(device=device, dtype=dtype)
        self.aux_mean = aux_mean.to(device=device, dtype=dtype)
        self.aux_std = aux_std.to(device=device, dtype=dtype)
        self.g_obs_mean = g_obs_mean.to(device=device, dtype=dtype)
        self.g_obs_std = g_obs_std.to(device=device, dtype=dtype)
        self.k_warmup = int(k_warmup)
        self.n_steps = int(n_steps)
        self.target_start = int(k_warmup if target_start is None else target_start)
        self.target_end = int(n_steps if target_end is None else target_end)
        self.device = device
        self.dtype = dtype
        self.c_pool_norm = c_pool_norm
        self.c_pool_mean = (
            c_pool_mean.to(device=device, dtype=dtype)
            if c_pool_mean is not None
            else None
        )
        self.c_pool_std = (
            c_pool_std.to(device=device, dtype=dtype)
            if c_pool_std is not None
            else None
        )
        # FEI knobs. "off" = legacy v5 (Haar norms only). "replace" = feed
        # FEI into the g_obs slot (head architecture unchanged; g_obs_mean/std
        # already hold FEI stats). "concat" = feed both Haar norms and FEI
        # (head has a separate fei_k slot).
        self.fei_obs = fei_obs
        self.fei_k = int(fei_k)
        self.fei_mean = (
            fei_mean.to(device=device, dtype=dtype) if fei_mean is not None else None
        )
        self.fei_std = (
            fei_std.to(device=device, dtype=dtype) if fei_std is not None else None
        )
        self.is_active: bool = False
        self.c_pool: Optional[torch.Tensor] = None
        self.aux: Optional[torch.Tensor] = None
        self.g_obs_buf: list[float] = []
        self.fei_buf: list[float] = []
        # σ_low depends only on latent (H, W) and is constant across steps —
        # cached on first record_latent_pre_forward call so we don't recompute.
        self._sigma_low: Optional[float] = None
        self.alpha_eff: float = 0.0
        self.gain: float = 1.0
        self.baseline_lambda: float = float(baseline_lambda)

    @classmethod
    def from_safetensors(
        cls, path: str | Path, *, device: torch.device
    ) -> "OnlineDCWCalibrator":
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        schema = meta.get("schema")
        if schema not in _VALID_SCHEMAS:
            raise ValueError(
                f"{path}: unexpected schema {schema!r}, expected one of "
                f"{_VALID_SCHEMAS}"
            )
        # Pre-lambda_scalar v4 artifacts default to alpha_residual. The new
        # controller only does lambda_scalar — refuse to silently misinterpret.
        target_kind = meta.get("target_kind", "lambda_scalar")
        if target_kind != "lambda_scalar":
            raise ValueError(
                f"{path}: target_kind={target_kind!r} is no longer supported. "
                "Either retrain with the current trainer (always lambda_scalar) "
                "or `git checkout` to the pre-cleanup controller for compat."
            )
        k_warmup = int(meta.get("k_warmup", 7))
        n_steps = int(meta.get("n_steps", 28))
        target_start = int(meta.get("target_start", k_warmup))
        target_end = int(meta.get("target_end", n_steps))
        # Legacy artifacts (pre-baseline) → 0.0 = no scalar baseline applied,
        # exactly the previous behavior.
        baseline_lambda = float(meta.get("baseline_lambda", 0.0))
        fei_obs = meta.get("fei_obs", "off")
        if fei_obs not in ("off", "replace", "concat"):
            raise ValueError(
                f"{path}: unknown fei_obs={fei_obs!r} in metadata. "
                "Expected one of off/replace/concat."
            )
        # Schema vs fei_obs consistency check — catches hand-edited artifacts
        # that would silently bypass the FEI capture path.
        expected_schema = {
            "off": "dcw_v5_lambda_scalar",
            "replace": "dcw_v6_fei_replace",
            "concat": "dcw_v6_fei_concat",
        }[fei_obs]
        if schema != expected_schema and schema != "dcw_v4_fusion_head":
            raise ValueError(
                f"{path}: schema={schema!r} disagrees with fei_obs={fei_obs!r} "
                f"(expected {expected_schema!r}). Retrain."
            )
        fei_k = int(meta.get("fei_k", 0))
        if fei_obs == "concat" and fei_k <= 0:
            raise ValueError(
                f"{path}: fei_obs=concat requires fei_k>0 in metadata; got {fei_k}."
            )
        if fei_obs != "concat" and fei_k != 0:
            # Trainer sets fei_k=0 outside concat mode; defensive.
            fei_k = 0

        head_sd = {
            k[len("head.") :]: v for k, v in tensors.items() if k.startswith("head.")
        }
        if "alpha_mlp.0.weight" not in head_sd:
            raise ValueError(
                f"{path}: missing 'head.alpha_mlp.*' keys — artifact predates "
                "the alpha/sigma trunk split. Retrain with `make dcw-train`."
            )
        in_dim = int(head_sd["alpha_mlp.0.weight"].shape[0])
        # Post-cleanup FusionHead in_dim = cat_dim + k + aux_dim. Old v4/v5
        # artifacts have an extra `aspect_emb_dim` (=16 by default) of phantom
        # slots in alpha_mlp's input — load_state_dict will fail with a clear
        # shape mismatch on alpha_mlp.1.weight, prompting a retrain.
        if "aspect_emb.weight" in head_sd:
            raise ValueError(
                f"{path}: artifact contains 'aspect_emb.weight' — predates the "
                "bucket-cosmetic removal. Retrain with `make dcw-train`."
            )
        aux_dim = 3
        if "c_proj.1.weight" in head_sd:
            c_proj_w = head_sd["c_proj.1.weight"]
            c_proj_dim = int(c_proj_w.shape[0])
            c_pool_dim = int(c_proj_w.shape[1])
            cat_dim = c_proj_dim
        else:
            c_proj_dim = 0
            c_pool_dim = in_dim - (k_warmup + fei_k + aux_dim)
            cat_dim = c_pool_dim
        if cat_dim + k_warmup + fei_k + aux_dim != in_dim:
            raise ValueError(
                f"{path}: shape mismatch — cat({cat_dim}) + k({k_warmup}) "
                f"+ fei_k({fei_k}) + aux({aux_dim}) != alpha_mlp.0 in_dim({in_dim}). "
                "Likely a pre-cleanup artifact with aspect_emb slots; retrain."
            )
        head = FusionHead(
            c_pool_dim=c_pool_dim,
            k=k_warmup,
            aux_dim=aux_dim,
            c_proj_dim=c_proj_dim,
            fei_k=fei_k,
        )
        # sigma_mlp keys are stripped at save time (σ̂² path is gone), so
        # load with strict=False to tolerate their absence.
        head.load_state_dict(head_sd, strict=False)

        c_pool_norm = meta.get("c_pool_norm", "none")
        if c_pool_norm not in ("none", "l2", "standardize", "l2_then_standardize"):
            raise ValueError(
                f"{path}: unknown c_pool_norm={c_pool_norm!r}. "
                "Either retrain with the current trainer or update the calibrator."
            )
        ctrl = cls(
            head=head,
            centroid=tensors["centroid_c_pool"],
            aux_mean=tensors["aux_mean"],
            aux_std=tensors["aux_std"],
            g_obs_mean=tensors["g_obs_mean"],
            g_obs_std=tensors["g_obs_std"],
            k_warmup=k_warmup,
            n_steps=n_steps,
            device=device,
            target_start=target_start,
            target_end=target_end,
            c_pool_norm=c_pool_norm,
            c_pool_mean=tensors.get("c_pool_mean"),
            c_pool_std=tensors.get("c_pool_std"),
            baseline_lambda=baseline_lambda,
            fei_obs=fei_obs,
            fei_k=fei_k,
            fei_mean=tensors.get("fei_mean"),
            fei_std=tensors.get("fei_std"),
        )
        logger.info(
            "DCW calibrator: loaded %s (schema=%s, k=%d, target=[%d:%d], "
            "%d steps, c_pool_norm=%s, baseline_lambda=%+.4g, fei_obs=%s, fei_k=%d)",
            path.name,
            schema,
            k_warmup,
            target_start,
            target_end,
            n_steps,
            c_pool_norm,
            baseline_lambda,
            fei_obs,
            fei_k,
        )
        return ctrl

    def setup(
        self,
        embed: torch.Tensor,
        embed_mask: Optional[torch.Tensor],
        *,
        gain: float = 1.0,
    ) -> None:
        """Compute c_pool + aux for this generation. Idempotent."""
        self.is_active = False
        self.g_obs_buf = []
        self.fei_buf = []
        self._sigma_low = None
        self.alpha_eff = 0.0
        self.gain = float(gain)

        # Pool the first batch row's embed (single-prompt assumption — matches
        # the trainer's per-prompt format). embed: (B, L, 1024).
        e = embed[0].to(self.device, dtype=self.dtype)
        if embed_mask is not None:
            mask = embed_mask[0].to(self.device, dtype=torch.bool)
            valid = e[mask]
            cap_len = int(mask.sum().item())
        else:
            valid = e
            cap_len = e.shape[0]
        if valid.numel() == 0:
            logger.warning("DCW calibrator: empty embed mask — disabling")
            return

        c_pool_raw = valid.mean(dim=0)
        token_l2 = valid.norm(dim=-1)
        # cos_centroid stays raw — the trainer's centroid was computed on raw
        # c_pool, and the cos itself is the aux feature, not the head input.
        cos_centroid = float(
            torch.dot(c_pool_raw, self.centroid)
            / (c_pool_raw.norm() * self.centroid.norm() + 1e-9)
        )
        aux_raw = torch.tensor(
            [float(cap_len), cos_centroid, float(token_l2.std().item())],
            device=self.device,
            dtype=self.dtype,
        )
        # Apply the same preprocessing the trainer used to the head's c_pool input.
        c_pool = c_pool_raw
        if self.c_pool_norm in ("l2", "l2_then_standardize"):
            c_pool = c_pool / (c_pool.norm() + 1e-9)
        if self.c_pool_norm in ("standardize", "l2_then_standardize"):
            if self.c_pool_mean is None or self.c_pool_std is None:
                raise RuntimeError(
                    "c_pool_norm requests standardize but artifact has no "
                    "c_pool_mean / c_pool_std tensors — retrain to ship them."
                )
            c_pool = (c_pool - self.c_pool_mean) / self.c_pool_std
        self.c_pool = c_pool
        self.aux = (aux_raw - self.aux_mean) / self.aux_std
        self.is_active = True
        logger.info(
            "DCW calibrator: setup target=[%d:%d] gain=%.4g baseline=%+.4g "
            "cap_len=%d cos_centroid=%.3f c_pool_norm=%s",
            self.target_start,
            self.target_end,
            self.gain,
            self.baseline_lambda,
            cap_len,
            cos_centroid,
            self.c_pool_norm,
        )

    def record(self, step_i: int, noise_pred: torch.Tensor) -> None:
        """Observe LL-band norm of the post-CFG velocity at warmup steps.

        In ``fei_obs="replace"`` mode the head's g_obs slot is fed FEI from
        the pre-forward latent (see ``record_latent_pre_forward``), so this
        no-ops to avoid populating an unused buffer.
        """
        if not self.is_active or step_i >= self.k_warmup:
            return
        if self.fei_obs == "replace":
            return
        self.g_obs_buf.append(haar_LL_norm(noise_pred))

    def record_latent_pre_forward(self, step_i: int, latents: torch.Tensor) -> None:
        """Capture 2-band FEI low-band energy on the latent entering this step.

        Must be called *before* the model forward at warmup steps when
        ``fei_obs != "off"`` — timing must match
        ``scripts/dcw/trajectory.py`` (pre-forward, pre-DCW correction).
        No-ops when inactive, past warmup, or in legacy ``fei_obs="off"`` mode.

        ``latents`` is the same tensor passed to ``anima(...)``: shape
        ``(B, C, T, H, W)`` on Anima (the T axis is squeezed before the FEI
        compute, matching the data-collection call site).
        """
        if not self.is_active or step_i >= self.k_warmup:
            return
        if self.fei_obs == "off":
            return
        z = latents[:, :, 0] if latents.ndim == 5 else latents
        h_lat, w_lat = z.shape[-2], z.shape[-1]
        if self._sigma_low is None:
            self._sigma_low = fei_sigma_low(h_lat, w_lat, _FEI_SIGMA_LOW_DIV)
        # Single-prompt assumption — pool's row 0 only, same as g_obs/c_pool.
        fei = compute_fei_2band(z[:1], self._sigma_low)  # (1, 2) fp32
        self.fei_buf.append(float(fei[0, 0].item()))

    def fire_head_if_due(self, step_i: int) -> None:
        """Run the MLP at i == k_warmup. Sets self.alpha_eff for the tail."""
        if not self.is_active or step_i != self.k_warmup:
            return

        # Source-buffer integrity checks vary by mode: replace feeds the g_obs
        # slot from FEI; concat needs both; off needs only Haar norms.
        if self.fei_obs == "replace":
            if len(self.fei_buf) < self.k_warmup:
                logger.warning(
                    "DCW calibrator: fei_obs=replace expects %d FEI obs, got %d "
                    "— did the caller forget record_latent_pre_forward? Disabling.",
                    self.k_warmup,
                    len(self.fei_buf),
                )
                self.alpha_eff = 0.0
                return
        else:
            if len(self.g_obs_buf) < self.k_warmup:
                logger.warning(
                    "DCW calibrator: only %d/%d warmup obs collected — disabling",
                    len(self.g_obs_buf),
                    self.k_warmup,
                )
                self.alpha_eff = 0.0
                return
            if self.fei_obs == "concat" and len(self.fei_buf) < self.k_warmup:
                logger.warning(
                    "DCW calibrator: fei_obs=concat expects %d FEI obs, got %d "
                    "— did the caller forget record_latent_pre_forward? Disabling.",
                    self.k_warmup,
                    len(self.fei_buf),
                )
                self.alpha_eff = 0.0
                return

        if self.fei_obs == "replace":
            # The head's g_obs slot carries FEI. g_obs_mean/std were redirected
            # to FEI stats at training time, so the same normalization formula
            # works as a drop-in replacement.
            fei = torch.tensor(
                self.fei_buf[: self.k_warmup], device=self.device, dtype=self.dtype
            )
            head_g_obs = (fei - self.g_obs_mean) / self.g_obs_std
            head_fei: Optional[torch.Tensor] = None
        else:
            g_obs = torch.tensor(
                self.g_obs_buf[: self.k_warmup], device=self.device, dtype=self.dtype
            )
            head_g_obs = (g_obs - self.g_obs_mean) / self.g_obs_std
            if self.fei_obs == "concat":
                fei = torch.tensor(
                    self.fei_buf[: self.k_warmup], device=self.device, dtype=self.dtype
                )
                if self.fei_mean is None or self.fei_std is None:
                    raise RuntimeError(
                        "fei_obs=concat requires fei_mean/fei_std in the artifact."
                    )
                head_fei = ((fei - self.fei_mean) / self.fei_std).unsqueeze(0)
            else:
                head_fei = None

        with torch.no_grad():
            alpha_hat, _ = self.head(
                self.c_pool.unsqueeze(0),
                head_g_obs.unsqueeze(0),
                self.aux.unsqueeze(0),
                fei=head_fei,
            )
        self.alpha_eff = float(alpha_hat[0].item())
        logger.info(
            "DCW calibrator: head fired at step %d — α̂=%+.4g (fei_obs=%s)",
            step_i,
            self.alpha_eff,
            self.fei_obs,
        )

    def lambda_for_step(self, step_i: int, sigma_i: float) -> float:
        """Per-step λ for ``apply_dcw(..., schedule='const', lam=λ_i)``.

        Baseline applies on every step (matches the data-collection
        scalar — no warmup dead zone). The residual α̂·gain term only
        contributes once the head has fired, inside [target_start, target_end).
        """
        if not self.is_active:
            return 0.0
        env = 1.0 - sigma_i
        lam_i = self.baseline_lambda * env
        if self.target_start <= step_i < self.target_end:
            lam_i += self.alpha_eff * self.gain * env
        return max(-_LAMBDA_CLAMP, min(_LAMBDA_CLAMP, lam_i))
