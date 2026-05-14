"""Training backend protocol and shared types.

Every concrete backend (Kohya, Diffusers, ...) implements `TrainingBackend`.
The orchestrator and CLI interact exclusively through this interface, so
backends can be swapped or run in isolated subprocesses without coupling.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lorahub.core.events import TrainingEvent


class ModelArch(enum.StrEnum):
    sd15 = "sd15"
    sdxl = "sdxl"
    flux = "flux"
    sd3 = "sd3"


class Severity(enum.StrEnum):
    error = "error"
    warning = "warning"
    info = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class VRAMEstimate:
    """Estimated peak VRAM usage in MiB, broken down by component."""

    model_mib: int
    optimizer_mib: int
    activations_mib: int
    overhead_mib: int = 256

    @property
    def total_mib(self) -> int:
        return self.model_mib + self.optimizer_mib + self.activations_mib + self.overhead_mib

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024


@dataclass(slots=True)
class TrainingHandle:
    """Live handle to a running training process."""

    job_id: str
    pid: int | None = None
    _stop_fn: Callable[[bool], None] | None = field(default=None, repr=False)

    def stop(self, *, graceful: bool = True) -> None:
        if self._stop_fn is not None:
            self._stop_fn(graceful)


@runtime_checkable
class TrainingBackend(Protocol):
    """Contract that every training backend must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def supported_archs(self) -> set[ModelArch]: ...

    def validate(self, cfg: dict[str, Any]) -> list[ValidationIssue]:
        """Check config for errors before launching. `cfg` is the raw recipe dict."""
        ...

    def estimate_vram(self, cfg: dict[str, Any]) -> VRAMEstimate:
        """Estimate peak VRAM usage for the given config."""
        ...

    def launch(
        self,
        cfg: dict[str, Any],
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
    ) -> TrainingHandle:
        """Start training. Returns immediately with a handle.

        `workspace` is a directory where the backend writes checkpoints,
        samples, and logs. `on_event` is called from a background thread
        whenever the backend has something to report.
        """
        ...
