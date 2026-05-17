"""Subprocess-side monkey-patch that swaps kohya's FA2 dispatcher for FA3/FA4.

kohya sd-scripts only ships FlashAttention 2 wiring today (`--attn_mode flash`
loads `flash_attn.flash_attn_func`). FA3 and FA4 expose newer kernels via the
`flash_attn_interface` and `flash_attn_4` packages but use a different module
path, so getting kohya to call them requires either an upstream patch or an
in-process shim. This module is the shim.

How it loads:
    The kohya backend's runner stages this patch via PYTHONSTARTUP — every
    Python interpreter the backend spawns runs a tiny `sitecustomize.py`
    (written into the workspace and exposed via PYTHONPATH) that calls
    :func:`apply`. Behaviour is gated on
    ``LORAHUB_KOHYA_ATTN_OVERRIDE``: when unset / empty, this module is a
    no-op so the host venv stays unaffected.

Failure mode:
    Anything raised inside the patch is caught and logged but never crashes
    the trainer. Worst case the user keeps their FA2 path — the exact
    fall-back kohya already provides — instead of getting FA3/FA4 speed.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

log = logging.getLogger(__name__)

# Env var the kohya compiler stamps into the spawn environment when the
# user picks `attention.training=flash3` or `flash4`.
OVERRIDE_ENV = "LORAHUB_KOHYA_ATTN_OVERRIDE"

# Modules inside sd-scripts that hold the attention dispatcher. We try every
# name because the file lives under different package paths depending on the
# kohya install layout.
_TARGET_MODULES = (
    "library.attention",
    "library.flash_attn_compat",
    "sd_scripts.library.attention",
)


def _select_dispatcher(target: str) -> Any | None:
    """Return the FA3 / FA4 attention callable, or None if the wheel is missing.

    We look up the dispatcher lazily so a host that doesn't have the wheel
    installed simply keeps kohya's FA2 path. The ``flash_attn_interface``
    package is the FA3 distribution (Hopper-only); FA4 ships under
    ``flash_attn_4`` (or a 4.x ``flash_attn`` fork). Both expose a
    ``flash_attn_func`` symbol with an FA2-compatible signature, which is
    what kohya consumes.
    """
    if target == "flash3":
        try:
            import flash_attn_interface  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            log.warning("FA3 requested but flash_attn_interface is not importable: %s", exc)
            return None
        return getattr(flash_attn_interface, "flash_attn_func", None)
    if target == "flash4":
        for mod_name in ("flash_attn_4", "flash_attn"):
            try:
                mod = __import__(mod_name)
            except Exception:  # noqa: BLE001
                continue
            fn = getattr(mod, "flash_attn_func", None)
            if fn is None:
                continue
            if mod_name == "flash_attn_4":
                return fn
            version = getattr(mod, "__version__", "") or ""
            if version.startswith("4"):
                return fn
        log.warning("FA4 requested but no FA4 wheel is importable")
        return None
    return None


def _patch_module(module: Any, dispatcher: Any, target: str) -> None:
    """Replace the attention callable on a kohya attention module."""
    for name in ("flash_attn_func", "_flash_attn_func", "attn_func"):
        if hasattr(module, name):
            setattr(module, name, dispatcher)
    setattr(module, "__lorahub_attn_override__", target)


def apply() -> bool:
    """Install the FA3/FA4 dispatcher into kohya's attention module.

    Returns True when a patch was scheduled, False otherwise. Safe to call
    from a startup hook: any failure (missing wheel, unknown override
    value) is logged but never raises.
    """
    target = (os.environ.get(OVERRIDE_ENV) or "").strip()
    if not target:
        return False
    if target not in ("flash3", "flash4"):
        log.warning("Unknown %s value %r; expected flash3 or flash4", OVERRIDE_ENV, target)
        return False

    dispatcher = _select_dispatcher(target)
    if dispatcher is None:
        return False

    # Patch modules that are already imported (e.g. when this is invoked
    # mid-process by tests). For modules imported later, register a finder
    # that patches them at first import.
    for name in _TARGET_MODULES:
        mod = sys.modules.get(name)
        if mod is not None:
            try:
                _patch_module(mod, dispatcher, target)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to patch %s: %s", name, exc)

    sys.meta_path.insert(0, _DeferredAttnPatcher(dispatcher, target))
    log.info("LoraHub kohya attention override active: %s", target)
    return True


class _DeferredAttnPatcher:
    """Meta-path finder that patches kohya attention modules post-import.

    We never claim to load anything — :py:meth:`find_spec` returns ``None``.
    But our presence on ``sys.meta_path`` means we receive a callback for
    every import attempt, which gives us a chance to walk ``sys.modules``
    after the real loader has populated it. Cheap, and survives kohya's
    ``importlib.reload`` shenanigans during cache warmup.
    """

    __slots__ = ("_dispatcher", "_target", "_done")

    def __init__(self, dispatcher: Any, target: str) -> None:
        self._dispatcher = dispatcher
        self._target = target
        self._done: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Any = None,  # noqa: ARG002
        target: Any = None,  # noqa: ARG002
    ) -> None:
        if fullname in _TARGET_MODULES and fullname not in self._done:
            mod = sys.modules.get(fullname)
            if mod is not None:
                try:
                    _patch_module(mod, self._dispatcher, self._target)
                    self._done.add(fullname)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to patch %s: %s", fullname, exc)
        return None


__all__ = ["OVERRIDE_ENV", "apply"]
