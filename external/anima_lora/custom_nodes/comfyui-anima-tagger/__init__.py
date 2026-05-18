"""Anima Tagger ComfyUI custom nodes.

Two nodes:

* ``AnimaTaggerLoader`` - loads an AnimaTagger checkpoint from disk and
  emits a reusable ``ANIMA_TAGGER`` socket. ComfyUI memoizes loader
  outputs so the tagger persists across graph runs.
* ``AnimaTaggerCaption`` - takes an ``ANIMA_TAGGER`` + ``IMAGE`` and emits
  the predicted caption as a ``STRING``. Drop-in for any STRING input
  (CLIPTextEncode, AnimaDirectEdit's ``prompt_src_override``, etc.).

The loader's ``ANIMA_TAGGER`` socket is also consumed by ``AnimaDirectEdit``
in the ``comfyui-anima-directedit`` package - install both packages to use
DirectEdit with on-image tagger captioning.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
