"""Shared exception types for training backends.

Both `kohya` and `diffusion_pipe` raise the same `BootstrapError` so the
bootstrap session and routers can catch a single class regardless of which
backend the user is installing or resolving.

The class supports two construction styles, kept compatible with both
historical call sites in the kohya backend:

    raise BootstrapError("free-form remediation message")
    raise BootstrapError("clone", returncode=128)
"""

from __future__ import annotations


class BootstrapError(RuntimeError):
    """Raised when a backend cannot be located, validated, or installed."""

    def __init__(
        self,
        message_or_step: str,
        returncode: int | None = None,
    ) -> None:
        if returncode is not None:
            # Installer-style: a single subprocess step failed.
            self.step: str | None = message_or_step
            self.returncode: int | None = returncode
            super().__init__(
                f"step {message_or_step!r} failed (exit code {returncode})"
            )
        else:
            self.step = None
            self.returncode = None
            super().__init__(message_or_step)


__all__ = ["BootstrapError"]
