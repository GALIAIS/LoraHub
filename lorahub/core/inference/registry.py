"""Registry of preview inference backends.

LoRA preview rendering used to be hardcoded to Anima — when a recipe
selected a different arch (Flux/Wan/HunyuanVideo) the worker silently
fell back to ``StubInference`` and only logged a warning. The registry
formalises the dispatch so we can:

  * route by ``cfg.base_model.arch`` to the right backend (Anima vs.
    a generic diffusers path) instead of hardcoding one;
  * surface a ``preview_unavailable`` event when no backend can serve
    the arch, so the UI shows *why* previews are placeholders;
  * lazy-import heavy deps (diffusers) so a missing wheel never blocks
    job launch — it just fails the ``is_available`` gate.

Design notes:
  * Backends are looked up via *factory functions* registered in import
    order. Each factory receives the (arch, recipe, workspace) tuple and
    returns either a configured ``InferenceBackend`` or ``None`` (meaning
    "I don't claim this arch / my prerequisites aren't here").
  * The registry doesn't enforce arch->backend uniqueness. A future cut
    can add multiple backends per arch family with priority ordering.
  * Backends register themselves at module import time. ``anima.py``
    imports this module and calls ``register_backend(...)``, so simply
    importing ``lorahub.core.inference`` (which the worker already does)
    populates the registry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lorahub.core.inference import PromptSpec

log = logging.getLogger(__name__)


@runtime_checkable
class InferenceBackend(Protocol):
    """Runtime renderer contract.

    Mirrors the historical ``AnimaInference`` Protocol so the existing
    ``PreviewWorker.render`` call-site works without changes. Backends
    expose ``name`` (for the ``preview_unavailable`` event payload and
    structured logs) and ``is_available(arch=...)`` so the registry can
    skip them cheaply.
    """

    name: str

    def is_available(self, *, arch: str) -> bool: ...

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None: ...


# A factory takes the lookup context (arch + recipe + workspace) and
# returns either a configured backend or None. ``recipe`` and
# ``workspace`` are kept ``Any`` / optional so a factory can ignore
# them when its config is purely env-based (e.g. diffusers with default
# pretrained ids). Concrete signature:
#
#     factory(arch=arch, recipe=recipe, workspace=workspace) -> InferenceBackend | None
#
# Factories are responsible for catching their own ImportError /
# FileNotFoundError style failures and returning None — the registry
# treats anything raised as "skip this backend, try the next".
BackendFactory = Callable[..., "InferenceBackend | None"]


_REGISTRY: list[tuple[str, BackendFactory]] = []


def register_backend(name: str, factory: BackendFactory) -> None:
    """Append ``factory`` to the resolution chain.

    Re-registering the same name replaces the existing entry in place
    (handy for tests that swap backends). Order is preserved otherwise.
    """
    for idx, (existing, _) in enumerate(_REGISTRY):
        if existing == name:
            _REGISTRY[idx] = (name, factory)
            return
    _REGISTRY.append((name, factory))


def unregister_backend(name: str) -> None:
    """Remove ``name`` from the registry. No-op if not registered.
    Tests use this to reset state between cases."""
    for idx, (existing, _) in enumerate(list(_REGISTRY)):
        if existing == name:
            _REGISTRY.pop(idx)
            return


def registered_backend_names() -> list[str]:
    """Names of all backends currently in the resolution chain.

    Surfaced in the ``preview_unavailable`` event so the UI can show
    *which* backends were considered when no match was found.
    """
    return [name for name, _ in _REGISTRY]


def resolve_backend(
    *,
    arch: str,
    recipe: Any | None = None,
    workspace: Path | None = None,
) -> InferenceBackend | None:
    """First registered backend that can serve ``arch`` for this recipe.

    Each factory is tried in registration order. The first one returning
    a non-None backend wins. Factory exceptions are caught + logged so a
    single broken backend can't poison the resolution chain.
    """
    for name, factory in list(_REGISTRY):
        try:
            backend = factory(arch=arch, recipe=recipe, workspace=workspace)
        except Exception:  # noqa: BLE001
            log.exception("inference backend factory %r raised; skipping", name)
            continue
        if backend is None:
            continue
        # Sanity: the factory should have already gated, but double-check
        # so a factory bug (returning a backend that lies about arch)
        # doesn't cascade into the worker.
        try:
            if not backend.is_available(arch=arch):
                continue
        except Exception:  # noqa: BLE001
            log.exception(
                "inference backend %r is_available(arch=%s) raised; skipping",
                name,
                arch,
            )
            continue
        return backend
    return None
