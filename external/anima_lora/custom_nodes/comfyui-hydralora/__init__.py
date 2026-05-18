"""Anima ComfyUI custom nodes.

Three single-purpose loader nodes; chain them via the MODEL socket when
a workflow needs more than one:

  - ``AnimaAdapterLoader`` — LoRA / HydraLoRA / ReFT (auto-detected
    from safetensors keys). HydraLoRA supports both σ-conditional and
    FeRA-style FEI-conditional live routing on the Hydra stack.
  - ``AnimaFeraLoader`` — author-faithful FeRA (Yin et al.,
    arXiv:2511.17979): global router on the latent's spectral energy +
    per-Linear stacked independent experts. Incompatible save format
    with the FEI-on-Hydra variant above; mutually exclusive with
    HydraLoRA-moe at load time.
  - ``AnimaPostfixLoader`` — prefix / postfix / cond context splicing
    (auto-detected from safetensors keys).

Adapter and postfix were a single toggle-bool node before v3.0.0; see
README §3.0.0 for the rationale on the split. ``AnimaFeraLoader`` was
added in v3.1.0.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
