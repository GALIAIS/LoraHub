from lorahub.api.settings import SettingsStore


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
