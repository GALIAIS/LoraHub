"""Image Studio API router package.

The original ``image_studio.py`` was a 2165-line single-file router covering
seven distinct responsibilities. It now lives as a package with one
sub-module per responsibility. This ``__init__`` re-exports an aggregate
``router`` so existing imports (``from lorahub.api.routers.image_studio
import router``) keep working.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here — that future
# import binds the name ``annotations`` to a ``_Feature`` object in this
# module's namespace, which would shadow the ``annotations`` sub-package
# below.

from fastapi import APIRouter

from . import ai as _ai
from . import annotations as _annotations
from . import audit as _audit
from . import captions as _captions
from . import curate as _curate
from . import datasets as _datasets
from . import dedupe as _dedupe
from . import intake as _intake
from . import library as _library
from . import listings as _listings
from . import ops as _ops
from . import ship as _ship
from . import similarity as _similarity
from . import tagging as _tagging

router = APIRouter()
router.include_router(_listings.router)
router.include_router(_annotations.router)
router.include_router(_ops.router)
router.include_router(_ai.router)
router.include_router(_audit.router)
router.include_router(_captions.router)
router.include_router(_curate.router)
router.include_router(_dedupe.router)
router.include_router(_datasets.router)
router.include_router(_intake.router)
router.include_router(_library.router)
router.include_router(_ship.router)
router.include_router(_similarity.router)
router.include_router(_tagging.router)

__all__ = ["router"]
