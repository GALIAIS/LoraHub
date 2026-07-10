from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from lorahub.api import terminal_runner
from lorahub.core.redaction import (
    REDACTED,
    redact_argv,
    redact_command_text,
    redact_data,
)


def test_redact_argv_masks_separate_and_inline_secret_options() -> None:
    argv = [
        "tool",
        "--token",
        "hf_abcdefghijklmnop",
        "--wandb_api_key=sk-abcdefghijklmnopqrstuvwxyz",
        "--tokenizer",
        "models/tokenizer",
    ]

    assert redact_argv(argv) == [
        "tool",
        "--token",
        REDACTED,
        f"--wandb_api_key={REDACTED}",
        "--tokenizer",
        "models/tokenizer",
    ]


def test_redact_argv_masks_shell_fragments_without_hiding_paths() -> None:
    command = (
        "curl -H 'Authorization: Bearer hf_abcdefghijklmnop' "
        "https://user:password@example.com/model -o C:/models/model.bin"
    )

    display = redact_argv(["cmd", "/c", command])

    assert "hf_abcdefghijklmnop" not in display[2]
    assert "password" not in display[2]
    assert display[2].count(REDACTED) == 2
    assert "C:/models/model.bin" in display[2]


def test_redact_command_text_masks_environment_and_bare_tokens() -> None:
    value = "HF_TOKEN=hf_abcdefghijklmnop ghp_abcdefghijklmnopqrstuvwxyz123456"

    redacted = redact_command_text(value)

    assert "hf_abcdefghijklmnop" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert redacted.count(REDACTED) == 2


def test_redact_data_masks_nested_secret_fields_and_cookies() -> None:
    value = {
        "headers": {"Authorization": "Bearer abc", "Cookie": "sid=secret"},
        "nested": [{"hf_token": "hf_abcdefghijklmnop"}],
        "message": "Set-Cookie: sid=secret; HttpOnly",
    }

    redacted = redact_data(value)

    assert redacted["headers"]["Authorization"] == REDACTED
    assert redacted["headers"]["Cookie"] == REDACTED
    assert redacted["nested"][0]["hf_token"] == REDACTED
    assert "sid=secret" not in redacted["message"]


def test_terminal_runner_redacts_argv_errors_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "hf_abcdefghijklmnop"

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(f"missing --token {secret}")

    monkeypatch.setattr(terminal_runner.subprocess, "Popen", fail_popen)
    caplog.set_level(logging.INFO, logger=terminal_runner.__name__)

    events = list(
        terminal_runner.stream_command(
            argv=["tool", "--token", secret],
            cwd=tmp_path,
            env={},
            timeout_s=5,
        )
    )

    combined = f"{events!r}\n{caplog.text}"
    assert secret not in combined
    assert REDACTED in combined


def test_terminal_runner_redacts_streamed_output(
    tmp_path: Path,
) -> None:
    secret = "hf_abcdefghijklmnop"
    events = list(
        terminal_runner.stream_command(
            argv=[
                sys.executable,
                "-c",
                f"print('HF_TOKEN={secret}')",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=5,
        )
    )

    output = "\n".join(str(event.get("data", "")) for event in events)
    assert secret not in output
    assert REDACTED in output
