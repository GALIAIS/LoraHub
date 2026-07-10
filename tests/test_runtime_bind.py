from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.api import runtime_bind


def _state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_bind, "user_state_path", lambda *_args: tmp_path)


def test_runtime_bind_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _state_root(tmp_path, monkeypatch)

    runtime_bind.write_runtime_bind("0.0.0.0", 18765, pid=1234)

    assert runtime_bind.read_runtime_bind() == runtime_bind.RuntimeBind(
        host="0.0.0.0",
        port=18765,
        pid=1234,
    )


def test_runtime_bind_rejects_linked_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    outside.write_text("preserve", encoding="utf-8")
    link = tmp_path / "uvicorn.bind.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(RuntimeError, match="linked runtime state"):
        runtime_bind.write_runtime_bind("127.0.0.1", 18765, pid=1)

    assert outside.read_text(encoding="utf-8") == "preserve"
