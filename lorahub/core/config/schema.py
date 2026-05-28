"""Backwards-compatible alias for ``lorahub.core.config``."""
from lorahub.core.config import *  # noqa: F401, F403
from lorahub.core.config import __all__ as _all
__all__ = list(_all)
