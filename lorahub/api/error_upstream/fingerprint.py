"""Stable error fingerprint.

Same crash → same hex string → same GitLab issue. We hash the things
that identify the *failure*, not the noise that varies between
occurrences (timestamps, request ids, line numbers in the leaf
frame). The fingerprint is also written back into the local store so
the dispatcher can collapse repeat reports into one upstream issue
without re-walking the stack.

Inputs the hash digests, in priority order:
    1. ``source`` + ``category`` — coarse bucket, always available.
    2. The first stack frame's *function name + file basename*. Line
       numbers are intentionally dropped: a one-line edit anywhere in
       the function changes the absolute line number even though the
       crash is the same.
    3. The first 200 chars of ``message``, with everything that looks
       like a uuid / hex id / number stripped so a unique id in the
       message doesn't bust the cache.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

from lorahub.api.error_reports import ErrorReport

# Strip noise that varies across occurrences but doesn't change the
# kind of failure: bare numbers, uuid-shaped tokens, hex addresses,
# memory addresses (``0x7fff…``), uuid hyphenated forms.
_NOISE_RE = re.compile(
    r"(?:0x[0-9a-fA-F]+"
    r"|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{32,}"
    r"|\d+)",
)

# Match a typical Python traceback frame:
#   File "C:\\path\\to\\thing.py", line 42, in some_function
_FRAME_RE = re.compile(
    r'File\s+"([^"]+)"[^,]*,\s*line\s*\d+,\s*in\s+([A-Za-z_][\w.<>]*)'
)


def _normalise(text: str) -> str:
    return _NOISE_RE.sub("#", text)


def _first_frame(stack: str | None) -> str:
    """Return ``"{basename}::{func}"`` for the first matched frame, or
    empty string if none. Picks the *first* frame because that's the
    code the user actually wrote — the bottom of the stack is usually
    in third-party libraries shared by every failure mode."""
    if not stack:
        return ""
    m = _FRAME_RE.search(stack)
    if m is None:
        return ""
    path, func = m.group(1), m.group(2)
    base = os.path.basename(path.replace("\\", "/"))
    return f"{base}::{func}"


def compute_fingerprint(report: ErrorReport, *, parts: Iterable[str] | None = None) -> str:
    """Deterministic 16-char hex digest of the report's identity.

    Pure function — no I/O, no clock reads. The 16-char prefix is
    plenty for de-dupe across a single repo (collision probability is
    1e-9 even at 100 000 reports) and is short enough to fit in an
    issue title prefix without crowding it.
    """
    bits: list[str] = list(parts or [])
    bits.append(report.source)
    bits.append(report.category)
    bits.append(_first_frame(report.stack))
    bits.append(_normalise((report.message or "")[:200]))
    payload = "\x1f".join(bit.strip() for bit in bits)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["compute_fingerprint"]
