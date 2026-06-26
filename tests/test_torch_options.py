from __future__ import annotations

from types import SimpleNamespace

import pytest

from lorahub.api import torch_options
from lorahub.api.torch_options import get_torch_options, supports_cuda


def test_driver_570_recommends_cuda_128() -> None:
    payload = get_torch_options("570.124.04", "8.6")

    assert payload["max_cuda"] == "cu128"
    recommended = [row for row in payload["options"] if row["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["cuda"] == "cu128"


def test_driver_550_falls_back_to_cuda_124() -> None:
    payload = get_torch_options("550.54.15", "8.6")

    assert payload["max_cuda"] == "cu124"
    assert supports_cuda("550.54.15", "cu128") is False
    assert supports_cuda("550.54.15", "cu124") is True


def test_unknown_driver_keeps_options_selectable() -> None:
    payload = get_torch_options(None, None)

    assert payload["driver_version"] is None
    assert payload["options"][0]["compatible"] is True
    assert payload["options"][0]["recommended"] is True


def test_v100_skips_torch_wheels_that_require_sm75() -> None:
    payload = get_torch_options("570.124.04", "7.0")

    recommended = [row for row in payload["options"] if row["recommended"]]
    assert recommended[0]["cuda"] == "cu126"
    assert payload["options"][0]["compatible"] is False


def test_driver_probe_uses_hidden_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(torch_options, "_find_nvidia_smi", lambda: "nvidia-smi")

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="570.124.04\n", stderr="")

    monkeypatch.setattr(torch_options, "_run_hidden", fake_run)

    assert torch_options.detect_nvidia_driver() == "570.124.04"
    assert calls[0][0] == "nvidia-smi"
