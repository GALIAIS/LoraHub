"""anima_lora backend module: third training backend.

`sorryhyun/anima_lora` is shipped vendored under `external/anima_lora/`.
This package re-exports the public surface; concrete implementations
live in submodules that mirror the kohya / dp backend layouts:

- :mod:`compiler` — TrainingConfig → argv translation (no file emit;
  upstream owns its own merge chain via base.toml / presets.toml /
  methods/<method>.toml).
- :mod:`bootstrap` (cut2) — vendored-copy detection + python path resolve.
- :mod:`backend` (cut2) — implements :class:`TrainingBackend`.
- :mod:`runner` (cut2) — subprocess launch + stdout → TrainingEvent parsing.
"""

from __future__ import annotations

from lorahub.core.backends.anima_lora.backend import AnimaLoraBackend
from lorahub.core.backends.anima_lora.compiler import (
    CompilationError,
    compile_config,
)

__all__ = ["AnimaLoraBackend", "CompilationError", "compile_config"]
