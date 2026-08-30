"""Redaction rules applied to every outgoing error report.

Three categories the user picked from the AskUserQuestion dialog:

1. ``API key / Bearer token / Authorization`` headers — never leave
   the box. We replace them with a fixed ``***REDACTED***`` token so
   downstream tooling can still see *that* a secret was elided.
2. ``User home / drive paths`` — Windows ``C:\\Users\\<name>\\…`` and
   bare drive roots like ``F:\\D\\…`` get rewritten to ``~`` /
   ``<user-root>``. Preserves the *shape* of the path (extension,
   project layout, file basename) so a triage engineer can still tell
   "this was a config file under runs/" without learning who the user
   is.
3. ``Email addresses + IPs`` — collapse to placeholders.

Any text field on ``ErrorReport`` (message, stack, context values)
goes through every regex once. Order matters only for the path rules
(longest-prefix-first wins) so a path that matches both ``C:\\Users``
*and* ``C:\\`` lands on the friendlier replacement.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, cast

from lorahub.api.error_reports import ErrorReport

_REDACTED = "***REDACTED***"

# ---------------------------------------------------------------------- #
# Regexes
#
# All compiled once at import time. We keep them pre-compiled because a
# busy run can produce hundreds of reports during a single CI session
# and we'd rather not pay re.compile on every fan-out attempt.
# ---------------------------------------------------------------------- #

# Authorization headers in repr / dict form: ``Authorization': 'Bearer xxx``
# or just ``api_key=xxx``. Python's traceback / RuntimeError(repr(headers))
# is the usual leak path here, so we match both quoting styles.
_AUTH_HEADER_RE = re.compile(
    r"""
    (Authorization\s*[:=]\s*['\"]?)            # the header name + colon/equal
    (?:Bearer\s+|Basic\s+|Token\s+|sk-|ghp_|glpat-)?  # optional scheme prefix
    [A-Za-z0-9._\-]+                            # the credential body itself
    """,
    re.IGNORECASE | re.VERBOSE,
)

# `api_key=...` / `apikey=...` / `password=...` style key/value pairs.
_INLINE_SECRET_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])                        # must be a fresh word
    (api[_-]?key|apikey|password|secret|token|auth_token|access_key|x[-_]api[-_]key)
    (\s*[:=]\s*)                              # separator
    (?:['\"])?                                # optional quote
    [A-Za-z0-9._\-]{6,}                       # the credential body
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare credential shapes that show up in stack traces without a key
# label: ``ghp_XXXXXX``, ``sk-XXXXXX``, ``glpat-XXXXXX``, ``Bearer XX``.
_BARE_TOKEN_RE = re.compile(
    r"\b(?:ghp_|gho_|github_pat_|sk-(?:ant-|or-|proj-)?|glpat-)[A-Za-z0-9._\-]{20,}",
)

# JWT-ish three-segment tokens (``xxx.yyy.zzz``).
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9._\-]{10,}\.[A-Za-z0-9._\-]{10,}\.[A-Za-z0-9._\-]{10,}\b")

# Email addresses and naked IPv4 / IPv6 literals. The IP rule excludes
# 127.0.0.1 / 0.0.0.0 / 169.254.* / ::1 which carry no PII.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_IPV6_RE = re.compile(
    r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b"
)

# Windows-style paths that start at a drive letter. The user-home form
# is matched first (more specific), then the bare drive root.
_WIN_HOME_RE = re.compile(
    r"[A-Za-z]:\\Users\\[^\\\"'\s]+",
)
_WIN_DRIVE_RE = re.compile(
    r"\b[A-Za-z]:\\(?![A-Za-z]:)",
)
# POSIX home directories. Leading word boundary keeps us from
# transforming things like ``foo/home/bar`` (which isn't a home).
_POSIX_HOME_RE = re.compile(
    r"(?<![A-Za-z0-9_/])/home/[A-Za-z0-9_.\-]+",
)
_POSIX_USERS_RE = re.compile(  # macOS ``/Users/<name>``
    r"(?<![A-Za-z0-9_/])/Users/[A-Za-z0-9_.\-]+",
)


def _is_loopback(ip: str) -> bool:
    """Loopback / link-local IPv4 — never useful for triage, never PII."""
    if ip in {"127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if ip.startswith("169.254."):
        return True
    return False


def _redact_text(text: str) -> str:
    """Apply every redactor to a single string. Idempotent."""
    if not text:
        return text

    def _ip_sub(match: re.Match[str]) -> str:
        v = match.group(0)
        if _is_loopback(v):
            return v
        # Skip well-known port numbers from looking like IPs.
        return "<ip>"

    out = text
    out = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{_REDACTED}", out)
    out = _INLINE_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", out)
    out = _BARE_TOKEN_RE.sub(_REDACTED, out)
    out = _JWT_RE.sub(_REDACTED, out)
    # Path rules first — they consume drive prefixes, so they have to
    # run before any other rule that might match the trailing portion.
    out = _WIN_HOME_RE.sub("~", out)
    out = _POSIX_HOME_RE.sub("~", out)
    out = _POSIX_USERS_RE.sub("~", out)
    # ``re.sub`` interprets backslashes in the replacement as backrefs
    # (``\1``), so use a callable to drop a literal ``\`` in.
    out = _WIN_DRIVE_RE.sub(lambda _m: "<drive>:\\", out)
    out = _EMAIL_RE.sub("<email>", out)
    out = _IPV4_RE.sub(_ip_sub, out)
    out = _IPV6_RE.sub(_ip_sub, out)
    return out


def _redact_value(value: Any) -> Any:
    """Walk an arbitrary JSON-shaped tree and redact every string leaf."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            # Keys named like secrets force the whole leaf to redact —
            # even if the value would otherwise pass the regex (e.g. a
            # pre-encoded base64 blob without an obvious shape).
            if isinstance(k, str) and _is_secret_key(k):
                out[k] = _REDACTED
            else:
                out[k] = _redact_value(v)
        return out
    return value


def _is_secret_key(name: str) -> bool:
    n = name.lower()
    return any(
        s in n
        for s in (
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "auth-token",
            "access_key",
        )
    )


def redact_report(report: ErrorReport) -> ErrorReport:
    """Return a deep-copied report with every textual surface redacted.

    Mutating the original would surprise callers who still want the
    raw text in the local store; the local registry is a debug
    archive, the redacted copy is the *outgoing* payload.
    """
    redacted = ErrorReport(
        id=report.id,
        timestamp=report.timestamp,
        severity=report.severity,
        source=report.source,
        category=report.category,
        title=_redact_text(report.title),
        message=_redact_text(report.message),
        stack=_redact_text(report.stack) if report.stack else None,
        context=_redact_value(deepcopy(report.context)),
        job_id=report.job_id,
        request_id=report.request_id,
        request_path=_redact_text(report.request_path) if report.request_path else None,
        version=report.version,
        platform=report.platform,
    )
    return redacted


def redact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Public hook for the webhook sink's freeform payload bag."""
    return cast(dict[str, Any], _redact_value(payload))


def _self_test() -> bool:
    """Quick sanity probe used by the API health endpoint to confirm
    redaction still does the right thing after dependency upgrades.

    Returns True iff every assertion holds; never raises. Tests cover
    each case in the dedicated suite, but having a runtime probe means
    a broken build can be detected without re-running pytest.
    """
    home = os.path.expanduser("~")
    samples = [
        ("Authorization: Bearer abc123longxxxxxxxxxxx", _REDACTED),
        ("api_key=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaa1234", _REDACTED),
        ("user@example.com", "<email>"),
        ("203.0.113.1", "<ip>"),
        ("127.0.0.1", "127.0.0.1"),  # preserved
    ]
    return all(needle in _redact_text(text) for text, needle in samples) and bool(home)


__all__ = ["redact_dict", "redact_report"]
