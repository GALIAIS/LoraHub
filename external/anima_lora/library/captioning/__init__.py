"""Image-to-caption helpers used by editing/inversion paths.

Ships :class:`AnimaTagger` — trained on the Anima caption distribution,
the ψ_src provider for DirectEdit when a checkpoint is present at
``models/captioners/anima-tagger-v1/``.

Exposes ``predict(pil_img)`` and ``predict_caption(pil_img)`` for a
comma-separated tag string.
"""

# AnimaTagger pulls in PE-Core lazily on import (it doesn't load weights
# until first predict), but the import itself touches torch/safetensors —
# keep it eager so that ``from library.captioning import AnimaTagger`` works
# but in environments without a built checkpoint, callers handle the
# ``FileNotFoundError`` from ``AnimaTagger.__init__``.
from library.captioning.anima_tagger import AnimaTagger

__all__ = ["AnimaTagger"]
