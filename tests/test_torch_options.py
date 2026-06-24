from __future__ import annotations

from lorahub.api.torch_options import get_torch_options, supports_cuda


def test_driver_570_recommends_cuda_128() -> None:
    payload = get_torch_options("570.124.04")

    assert payload["max_cuda"] == "cu128"
    recommended = [row for row in payload["options"] if row["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["cuda"] == "cu128"


def test_driver_550_falls_back_to_cuda_124() -> None:
    payload = get_torch_options("550.54.15")

    assert payload["max_cuda"] == "cu124"
    assert supports_cuda("550.54.15", "cu128") is False
    assert supports_cuda("550.54.15", "cu124") is True


def test_unknown_driver_keeps_options_selectable() -> None:
    payload = get_torch_options(None)

    assert payload["driver_version"] is None
    assert payload["options"][0]["compatible"] is True
    assert payload["options"][0]["recommended"] is True
