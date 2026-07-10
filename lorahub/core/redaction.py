"""Secret redaction for command previews and persisted process logs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

REDACTED = "***REDACTED***"

_SECRET_OPTION_NAMES = frozenset(
    {
        "access-key",
        "access-token",
        "api-key",
        "apikey",
        "auth-token",
        "authorization",
        "cookie",
        "github-token",
        "gitlab-token",
        "hf-token",
        "huggingface-hub-token",
        "modelscope-api-token",
        "password",
        "passwd",
        "private-key",
        "proxy-authorization",
        "secret",
        "set-cookie",
        "ssh-key",
        "token",
        "wandb-api-key",
    }
)

_SECRET_NAME_PATTERN = (
    r"(?:access[_-]key|access[_-]token|api[_-]?key|apikey|auth[_-]?token|"
    r"authorization|github[_-]token|gitlab[_-]token|hf[_-]token|"
    r"huggingface[_-]hub[_-]token|modelscope[_-]api[_-]token|password|passwd|"
    r"secret|token|wandb[_-]api[_-]key)"
)
_SECRET_KEY_VALUE_PATTERN = (
    r"(?:access[_-]key|access[_-]token|api[_-]?key|apikey|auth[_-]?token|"
    r"github[_-]token|gitlab[_-]token|hf[_-]token|"
    r"huggingface[_-]hub[_-]token|modelscope[_-]api[_-]token|password|passwd|"
    r"secret|token|wandb[_-]api[_-]key)"
)

_AUTHORIZATION_RE = re.compile(
    r"(?i)((?:proxy-)?authorization\s*[:=]\s*['\"]?)"
    r"(?:(bearer|basic|token)\s+([^\s,'\";]+)|([^\s,'\";]+))"
)
_CLI_SECRET_RE = re.compile(
    rf"(?i)(?<![\w-])(--{_SECRET_NAME_PATTERN})(?![\w-])"
    r"(\s*=\s*|\s+)(['\"]?)[^\s,'\";|&]+"
)
_KEY_VALUE_RE = re.compile(
    rf"(?i)(?<![\w-])({_SECRET_KEY_VALUE_PATTERN})(?![\w-])"
    r"(\s*[:=]\s*)(['\"]?)[^\s,'\";|&]+"
)
_URL_PASSWORD_RE = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://[^/@\s:]+:)([^/@\s]+)(@)"
)
_BARE_TOKEN_RE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{8,}|gh[oprsu]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,}|"
    r"sk-(?:ant-|or-|proj-)?[A-Za-z0-9_-]{20,})\b"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)
_COOKIE_RE = re.compile(r"(?im)((?:set-)?cookie\s*[:=]\s*)[^\r\n]+")


def _normalise_option_name(value: str) -> str:
    return value.lstrip("-").lower().replace("_", "-")


def redact_command_text(value: str) -> str:
    """Redact credentials embedded in a shell fragment without hiding paths."""
    if not value:
        return value

    redacted = _AUTHORIZATION_RE.sub(
        lambda match: (
            f"{match.group(1)}"
            f"{f'{match.group(2)} ' if match.group(2) else ''}"
            f"{REDACTED}"
        ),
        value,
    )
    redacted = _CLI_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}",
        redacted,
    )
    redacted = _KEY_VALUE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}",
        redacted,
    )
    redacted = _URL_PASSWORD_RE.sub(
        lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}",
        redacted,
    )
    redacted = _COOKIE_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    redacted = _BARE_TOKEN_RE.sub(REDACTED, redacted)
    return _JWT_RE.sub(REDACTED, redacted)


def redact_argv(argv: Iterable[str]) -> list[str]:
    """Return a display-safe argv while preserving its diagnostic structure."""
    values = [str(value) for value in argv]
    output: list[str] = []
    redact_next = False

    for value in values:
        if redact_next:
            output.append(REDACTED)
            redact_next = False
            continue

        option, separator, _option_value = value.partition("=")
        if option.startswith("-") and _normalise_option_name(option) in _SECRET_OPTION_NAMES:
            if separator:
                output.append(f"{option}={REDACTED}")
            else:
                output.append(option)
                redact_next = True
            continue

        output.append(redact_command_text(value))

    return output


def _is_secret_key(value: object) -> bool:
    name = _normalise_option_name(str(value))
    if name in _SECRET_OPTION_NAMES:
        return True
    suffixes = (
        "-api-key",
        "-auth-header",
        "-authorization",
        "-cookie",
        "-password",
        "-private-key",
        "-secret",
        "-ssh-key",
        "-token",
    )
    return name.endswith(suffixes)


def redact_data(value: Any) -> Any:
    """Recursively redact strings and values stored under secret-shaped keys."""
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret_key(key) else redact_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, set):
        return {redact_data(item) for item in value}
    if isinstance(value, str):
        return redact_command_text(value)
    return value


__all__ = ["REDACTED", "redact_argv", "redact_command_text", "redact_data"]
