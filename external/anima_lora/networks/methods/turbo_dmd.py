"""Turbo Anima — Decoupled DMD distillation harness.

Owns two plain ``LoRANetwork`` instances (student + fake) on one frozen Anima
DiT. Both call ``apply_to(unet)`` which chains them onto every targeted
Linear's forward — at runtime the chain order is::

    linear(x) -> fake.forward -> student.forward -> original_linear.forward

Each LoRA module short-circuits at ``not self.enabled`` (see
``lora_modules/lora.py::LoRAModule.forward``), so view-toggling is just
``set_enabled(bool)`` on each network — O(num_modules) Python loop, negligible
vs a DiT forward.

Used by ``scripts/distill_turbo.py``. Inference loads the saved
``anima_turbo.safetensors`` through the standard LoRA path (no inference-side
turbo code) — the student LoRA is just a normal LoRA with CFG=4 baked in.

Proposal: ``docs/proposal/turbo_anima_dmd_lora.md``.
Paper: Liu et al., "CFG Augmentation as the Spear, Distribution Matching as
the Shield" (arXiv:2511.22677).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from networks.lora_anima.factory import create_network
from networks.lora_anima.network import LoRANetwork

logger = logging.getLogger(__name__)

View = Literal["teacher", "student", "fake"]


class TurboDMDNetwork:
    """Two LoRA stacks on one frozen DiT, view-toggleable per forward.

    Not a ``nn.Module`` — it's a thin coordinator that holds two real
    ``LoRANetwork`` instances. The DiT itself is owned by the caller and
    stays frozen.
    """

    def __init__(
        self,
        unet,
        *,
        student_rank: int,
        fake_rank: int,
        student_alpha: float | None = None,
        fake_alpha: float | None = None,
        use_custom_down_autograd: bool = False,
    ) -> None:
        self.unet = unet
        self.student_rank = int(student_rank)
        self.fake_rank = int(fake_rank)

        # Plain LoRA on both — defaults from LoRANetworkCfg give us
        # use_moe_style=False / route_per_layer=False / router_source="none" /
        # use_ortho=False / use_timestep_mask=False / add_reft=False. No MoE,
        # no ortho, no T-LoRA, no ReFT — keep slice 1 KISS.
        # alpha = rank by default (scale = alpha/rank = 1.0) — matches the
        # project's LoRA-family convention. Halving alpha would silently halve
        # every student contribution per forward, making the 28→4 step trajectory
        # remap harder to bake without buying any stability we don't already
        # get from α-warmup + grad-clip + LR.
        # ``use_custom_down_autograd`` is forwarded as a ``**kwargs`` key because
        # ``create_network``'s positional surface doesn't include it — the factory
        # reads it out of ``kwargs`` and flips each module's flag post-construction.
        self.student: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.student_rank,
            network_alpha=student_alpha if student_alpha is not None else self.student_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
        )
        self.fake: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.fake_rank,
            network_alpha=fake_alpha if fake_alpha is not None else self.fake_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
        )

        # Apply order matters for the forward chain. We pick student-first so
        # the runtime chain is ``linear -> fake -> student -> original``. Both
        # are functionally symmetric (additive contributions) but having a
        # stable order makes debugging easier.
        self.student.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )
        self.fake.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )

        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), "
            f"fake rank={self.fake_rank} "
            f"({len(self.fake.unet_loras)} modules)"
        )

        # Start in teacher view — both off, base DiT is exactly itself.
        self._view: View = "teacher"
        self.set_view("teacher")

    # ----------------- view toggle -----------------

    # Per-view (student_on, fake_on) target states. Lookup avoids the
    # if/elif ladder and makes the "flip only what changed" diff explicit.
    _VIEW_FLAGS: dict[str, tuple[bool, bool]] = {
        "teacher": (False, False),
        "student": (True, False),
        "fake": (False, True),
    }

    def set_view(self, view: View) -> None:
        """Flip per-network enabled flags so the next DiT forward acts as
        the named view.

        - ``teacher``: both LoRA stacks off, DiT delivers base velocity.
        - ``student``: student on, fake off — produces v_student for x_pred.
        - ``fake``: fake on, student off — fake's score estimate at τ_DM.

        Short-circuits when already in the target view (consecutive teacher
        forwards in the CA + DM branches don't repay the ~O(num_modules)
        attribute-write loop, and dynamo doesn't get a chance to invalidate
        guards it would have re-validated anyway).
        """
        if view == self._view:
            return
        try:
            want_student, want_fake = self._VIEW_FLAGS[view]
        except KeyError as e:
            raise ValueError(
                f"Unknown view {view!r}; expected teacher/student/fake"
            ) from e
        cur_student, cur_fake = self._VIEW_FLAGS[self._view]
        if want_student != cur_student:
            self.student.set_enabled(want_student)
        if want_fake != cur_fake:
            self.fake.set_enabled(want_fake)
        self._view = view

    @property
    def view(self) -> View:
        return self._view

    # ----------------- param accessors -----------------

    def student_params(self):
        """Trainable params for the student optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]

    def fake_params(self):
        """Trainable params for the fake optimizer."""
        return [p for p in self.fake.parameters() if p.requires_grad]

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        Must be called AFTER both ``apply_to``'s — the LoRA networks add
        sub-modules to ``unet`` via ``add_module(lora.lora_name, lora)``, so
        a wholesale ``unet.requires_grad_(False)`` BEFORE apply would still
        be undone by the LoRA modules' own requires_grad=True params (good),
        but a wholesale call AFTER would zero those too (bad). We selectively
        walk only ``unet`` params whose name doesn't start with a LoRA prefix.
        """
        lora_prefixes = tuple(
            set(m.lora_name for m in self.student.unet_loras)
            | set(m.lora_name for m in self.fake.unet_loras)
        )
        n_frozen = 0
        for name, param in self.unet.named_parameters():
            if name.startswith(lora_prefixes):
                continue
            param.requires_grad_(False)
            n_frozen += 1
        logger.info(f"freeze_dit: {n_frozen} base params frozen")

    # ----------------- save / load -----------------

    def save_student(
        self,
        file: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Serialize only the student LoRA in the standard plain-LoRA layout.

        Output is loadable by ``inference.py --lora_weight <file>`` — the
        fake network is training scaffolding and never shipped.
        """
        from networks.lora_save import save_network_weights

        # Pull exactly the student's params, prefixed by LoRA-net key style
        # (this is what LoRANetwork.state_dict() returns — naturally so
        # because each LoRA was add_module'd onto the network).
        sd = self.student.state_dict()
        # Strip any non-LoRA keys defensively (router params, etc. — plain
        # LoRA shouldn't have any, but the LoRANetwork instance itself may
        # carry buffers that aren't load-bearing for inference).
        sd = {k: v for k, v in sd.items() if ".lora_" in k or ".alpha" in k}
        save_network_weights(
            sd,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant="standard",
        )
        logger.info(f"saved student LoRA → {file}  ({len(sd)} keys)")
