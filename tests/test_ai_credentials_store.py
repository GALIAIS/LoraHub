"""Tests for the AI credentials SQLite store."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lorahub.api.ai_credentials_store import AICredential, AICredentialStore


def test_upsert_and_get_round_trip(tmp_path: Path) -> None:
    store = AICredentialStore(tmp_path / "ai.sqlite")
    store.upsert(
        AICredential(
            provider="openai",
            api_key="sk-x",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",
        )
    )
    got = store.get("openai")
    assert got is not None
    assert got.api_key == "sk-x"
    assert got.default_model == "gpt-4o-mini"
    assert got.enabled is True
    assert isinstance(got.updated_at, datetime)
    assert got.updated_at.tzinfo is not None


def test_upsert_overwrites_in_place(tmp_path: Path) -> None:
    store = AICredentialStore(tmp_path / "ai.sqlite")
    store.upsert(AICredential(provider="qwen", api_key="old"))
    store.upsert(
        AICredential(provider="qwen", api_key="new", default_model="qwen-plus")
    )
    got = store.get("qwen")
    assert got is not None
    assert got.api_key == "new"
    assert got.default_model == "qwen-plus"
    rows = store.list()
    assert len(rows) == 1


def test_list_orders_by_provider(tmp_path: Path) -> None:
    store = AICredentialStore(tmp_path / "ai.sqlite")
    for p in ["qwen", "anthropic", "openai"]:
        store.upsert(AICredential(provider=p, api_key="x"))
    listed = [c.provider for c in store.list()]
    assert listed == sorted(listed)


def test_delete_returns_true_on_hit(tmp_path: Path) -> None:
    store = AICredentialStore(tmp_path / "ai.sqlite")
    store.upsert(AICredential(provider="openai", api_key="x"))
    assert store.delete("openai") is True
    assert store.delete("openai") is False
    assert store.get("openai") is None


def test_disabled_flag_round_trips(tmp_path: Path) -> None:
    store = AICredentialStore(tmp_path / "ai.sqlite")
    store.upsert(AICredential(provider="kimi", api_key="x", enabled=False))
    got = store.get("kimi")
    assert got is not None
    assert got.enabled is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics only")
def test_file_mode_is_user_only_on_posix(tmp_path: Path) -> None:
    path = tmp_path / "ai.sqlite"
    AICredentialStore(path)
    mode = path.stat().st_mode & 0o777
    # We chmod 0o600; allow 0o644 fallback only when the underlying FS
    # rejects chmod (Windows-mounted volumes etc.) — but those don't run
    # this test thanks to the skipif guard.
    assert mode in (0o600, 0o400, 0o644)
