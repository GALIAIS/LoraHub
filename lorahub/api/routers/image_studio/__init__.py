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

from .ai import router as ai_router
from .annotations import router as annotations_router
from .audit import router as audit_router
from .captions import router as captions_router
from .curate import router as curate_router
from .datasets import router as datasets_router
from .dedupe import router as dedupe_router
from .intake import router as intake_router
from .library import router as library_router
from .listings import router as listings_router
from .ops import router as ops_router
from .ship import router as ship_router
from .similarity import router as similarity_router
from .tagging import router as tagging_router

router = APIRouter()
router.include_router(listings_router)
router.include_router(annotations_router)
router.include_router(ops_router)
router.include_router(ai_router)
router.include_router(audit_router)
router.include_router(captions_router)
router.include_router(curate_router)
router.include_router(dedupe_router)
router.include_router(datasets_router)
router.include_router(intake_router)
router.include_router(library_router)
router.include_router(ship_router)
router.include_router(similarity_router)
router.include_router(tagging_router)

__all__ = ["router"]
