"""Sanity tests for the relaxed dirty-tree detection."""
from __future__ import annotations

from lorahub.api import system_update as su


def test_user_owned_paths_skip_dirty() -> None:
    assert su._is_user_owned_path("configs/anima_lora_default.yaml")
    assert su._is_user_owned_path("runs/abc/events.jsonl")
    assert su._is_user_owned_path(".env")
    assert su._is_user_owned_path(".env.local")
    assert su._is_user_owned_path("models/foo.safetensors")
    assert su._is_user_owned_path("external/anima_lora/uv.lock")
    assert su._is_user_owned_path("external/anima_lora/output/run42/cfg.yaml")


def test_source_paths_count_as_dirty() -> None:
    assert not su._is_user_owned_path("lorahub/api/app.py")
    assert not su._is_user_owned_path("lorahub/cli/service.py")
    assert not su._is_user_owned_path("scripts/install.sh")
    assert not su._is_user_owned_path("web/src/lib/api.ts")
    assert not su._is_user_owned_path("pyproject.toml")
