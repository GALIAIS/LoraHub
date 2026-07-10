from __future__ import annotations

import pytest
from fastapi import HTTPException

from lorahub.api.routers import network


def test_network_probe_rejects_non_http_url() -> None:
    with pytest.raises(HTTPException, match="http or https"):
        network.probe(network.ProbeRequest(urls=["file:///etc/passwd"]))


def test_network_probe_rejects_embedded_credentials() -> None:
    with pytest.raises(HTTPException, match="credentials"):
        network.probe(
            network.ProbeRequest(urls=["https://user:secret@example.com/"])
        )


def test_network_probe_deduplicates_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_probe(url: str, _timeout: float):  # type: ignore[no-untyped-def]
        calls.append(url)
        return 200, 1.0, None

    monkeypatch.setattr(network, "_probe_one", fake_probe)
    result = network.probe(
        network.ProbeRequest(
            urls=["https://example.com/", "https://example.com/"],
        )
    )

    assert len(result) == 1
    assert calls == ["https://example.com/"]
