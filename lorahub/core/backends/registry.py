"""Backend descriptor registry.

A tiny dict of metadata describing the training backends LoraHub knows
about. The HTTP API surfaces these to the UI so the user can pick a default
backend, see what's installed, and bootstrap one with one click.

Each backend keeps its own implementation under
`lorahub.core.backends.<id>` -- the registry only stores light metadata
plus references to the bootstrap installer and the backend class.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lorahub.core.backends.diffusion_pipe import bootstrap as dp_bootstrap
from lorahub.core.backends.diffusion_pipe import installer as dp_installer
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.backends.kohya import bootstrap as kohya_bootstrap
from lorahub.core.backends.kohya import installer as kohya_installer
from lorahub.core.backends.kohya.backend import KohyaBackend


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Light, stable metadata about one training backend.

    `bootstrap_func` and `backend_class` are stored as callables so callers
    don't have to reach back into `lorahub.core.backends.*` to wire things
    up. `default_path_func` is used by both probes and the bootstrap session
    to figure out where the backend lives if the user hasn't picked.
    """

    id: str
    name: str
    description: str
    repo_url: str
    default_path_func: Callable[[], Path]
    bootstrap_func: Callable[..., None]
    backend_class: type
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def default_path(self) -> Path:
        return self.default_path_func()


_BACKENDS: dict[str, BackendDescriptor] = {
    "kohya": BackendDescriptor(
        id="kohya",
        name="kohya-ss/sd-scripts",
        description=(
            "Battle-tested LoRA / DreamBooth trainer for SD1.5, SDXL, "
            "Flux and SD3."
        ),
        repo_url=kohya_installer.KOHYA_REPO_URL,
        default_path_func=kohya_bootstrap.default_sd_scripts_path,
        bootstrap_func=kohya_installer.bootstrap,
        backend_class=KohyaBackend,
    ),
    "diffusion-pipe": BackendDescriptor(
        id="diffusion-pipe",
        name="tdrussell/diffusion-pipe",
        description=(
            "DeepSpeed-based pipeline for image and video diffusion model "
            "fine-tuning. Scaffold only in v0.2 -- training launch ships in v0.3."
        ),
        repo_url=dp_installer.DIFFUSION_PIPE_REPO_URL,
        default_path_func=dp_bootstrap.default_repo_path,
        bootstrap_func=dp_installer.bootstrap,
        backend_class=DiffusionPipeBackend,
    ),
}


def list_backends() -> Iterable[BackendDescriptor]:
    """Yield descriptors in registration order."""
    return list(_BACKENDS.values())


def get_backend(backend_id: str) -> BackendDescriptor:
    """Look up a backend by id; raises KeyError if unknown."""
    if backend_id not in _BACKENDS:
        msg = f"unknown backend id: {backend_id!r}"
        raise KeyError(msg)
    return _BACKENDS[backend_id]


def known_ids() -> set[str]:
    return set(_BACKENDS.keys())


__all__ = [
    "BackendDescriptor",
    "get_backend",
    "known_ids",
    "list_backends",
]
