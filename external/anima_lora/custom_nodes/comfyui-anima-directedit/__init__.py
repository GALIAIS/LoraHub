"""Anima DirectEdit ComfyUI custom node.

One node:

* ``AnimaDirectEdit`` - takes an image plus two caption STRINGs
  (``source_tag`` describing the source, ``target_tag`` describing the
  edit; empty ``target_tag`` falls back to ``source_tag`` for a
  reconstruction sanity check) and invokes the DirectEdit invert +
  edit_forward primitives on the wired-in MODEL to produce an edited
  latent. Consumes ComfyUI's stock MODEL / CLIP / VAE sockets, so
  ``LoraLoader`` / ``comfyui-hydralora``'s adapter loader compose
  naturally upstream.

Both caption inputs are plain STRINGs — any node that emits STRING can
drive them.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
