import json
import os

import pytest

from lorahub.api.settings import Settings, SettingsStore


def test_settings_store_reads_github_proxy_env_alias(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("LORAHUB_GH_PROXY", "https://gh-proxy.example")

    settings = SettingsStore(tmp_path / "missing.json").load()

    assert settings.github_proxy == "https://gh-proxy.example"


def test_settings_store_prefers_canonical_github_proxy_env(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("LORAHUB_GH_PROXY", "https://legacy.example")
    monkeypatch.setenv("LORAHUB_GITHUB_PROXY", "https://canonical.example")

    settings = SettingsStore(tmp_path / "missing.json").load()

    assert settings.github_proxy == "https://canonical.example"


def test_settings_store_sanitizes_invalid_persisted_types(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "max_concurrent_jobs": "many",
                "terminal_unrestricted": "true",
                "gpu_dispatch_num_gpus": 99,
                "tagger_device": "metal",
                "huggingface_endpoint": ["https://invalid.example"],
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.max_concurrent_jobs == 1
    assert settings.terminal_unrestricted is False
    assert settings.gpu_dispatch_num_gpus is None
    assert settings.tagger_device == "auto"
    assert settings.huggingface_endpoint is None


def test_settings_store_preserves_nested_and_legacy_extra_values(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"extra": {"future": 1}, "legacy_option": "kept"}),
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.extra == {"future": 1, "legacy_option": "kept"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_settings_store_restricts_secret_file_permissions(tmp_path) -> None:
    path = tmp_path / "settings.json"

    SettingsStore(path).save(Settings(huggingface_token="secret"))

    assert path.stat().st_mode & 0o077 == 0
