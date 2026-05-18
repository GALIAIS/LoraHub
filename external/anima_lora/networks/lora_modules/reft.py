# LoReFT: Low-Rank Representation Fine-Tuning.
# Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)

from typing import Optional

import torch


class ReFTModule(torch.nn.Module):
    """LoReFT: low-rank residual-stream intervention.

        h_new = h + R^T(ΔW·h + b) * scale * multiplier

    R is an orthogonal rotation; ΔW (``learned_source``) is the learned delta
    in that subspace. Direct ΔW parameterisation (vs the paper's ``Wh + b − Rh``
    form) avoids activation-level cancellation, so the module runs in bf16
    without fp32 upcasts.

    ``org_module`` is normally a DiT Block — wrapping at the residual-stream
    level matches the paper (Wu et al., NeurIPS 2024 §3.3). Zero-init keeps
    delta=0 at step 0.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        embed_dim: Optional[int] = None,
        multiplier=1.0,
        reft_dim=4,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if embed_dim is None:
            if hasattr(org_module, "out_features"):
                embed_dim = org_module.out_features
            else:
                raise ValueError(
                    "embed_dim must be provided when wrapping a non-Linear module "
                    f"(got {type(org_module).__name__})"
                )
        self.reft_dim = reft_dim

        # R: orthogonal rotation into the intervention subspace.
        self.rotate_layer = torch.nn.Linear(embed_dim, reft_dim, bias=False)
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        r_rand = torch.randn(embed_dim, reft_dim, device=init_device)
        r_orth, _ = torch.linalg.qr(r_rand)
        self.rotate_layer.weight.data = r_orth.T.cpu().clone().contiguous()
        del r_rand, r_orth

        # ΔW within R's subspace; zero-init → delta=0 at step 0.
        self.learned_source = torch.nn.Linear(embed_dim, reft_dim)
        torch.nn.init.zeros_(self.learned_source.weight)
        torch.nn.init.zeros_(self.learned_source.bias)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = reft_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / reft_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        # See BaseLoRAModule._timestep_mask: all-ones default → identity, no
        # None-vs-Tensor guard. T-LoRA rebinds via set_reft_timestep_mask.
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, reft_dim, dtype=torch.float32),
            persistent=False,
        )

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, *args, **kwargs):
        # Works for wrapped Linear (x) and wrapped DiT Block (multi-arg).
        h = self.org_forward(*args, **kwargs)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return h

        delta = torch.nn.functional.linear(
            h, self.learned_source.weight, self.learned_source.bias
        )

        if self.training:
            delta = delta * self._timestep_mask

        if self.dropout is not None and self.training:
            delta = torch.nn.functional.dropout(delta, p=self.dropout)

        edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
        return h + edit * (self.multiplier * self.scale)

    def regularization(self):
        """||R R^T - I||^2."""
        R = self.rotate_layer.weight
        reg = torch.sum((R @ R.T - torch.eye(self.reft_dim, device=R.device)) ** 2)
        return reg
