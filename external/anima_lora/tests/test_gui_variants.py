"""Tests for the Track 2 GUI variant registry (``configs/gui-methods/*.toml``).

Built-in variants are discovered from each file's ``[variant]`` metadata
table (see ``gui/__init__.py``). This module pins:

* every built-in is loadable via ``load_method_preset(..., methods_subdir="gui-methods")``
* every built-in carries a ``[variant].family`` string
* no built-in surfaces a retired router knob as a TOML key
* output_name is unique across built-ins, with one explicit exception:
  the plain ``lora`` family ships three hardware/length variants that all
  bake to the same ``"anima"`` adapter on purpose.

Customs under ``configs/gui-methods/custom/`` are intentionally exempt —
users name and structure those freely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import toml

from library.config.io import load_method_preset

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_METHODS_DIR = REPO_ROOT / "configs" / "gui-methods"

# Retired in plan2 task #6 — surfacing them in a TOML must not silently
# resurrect the legacy code path. ``test_network_cfg.py`` already asserts the
# schema rejects them; here we just guarantee no built-in writes them.
_LEGACY_ROUTER_KEYS = ("use_hydra", "use_sigma_router", "use_fei_router")

# Built-in variants in the same family may legitimately share an output_name
# when they're hardware/length variants of the same model (e.g. all three
# plain-LoRA variants bake the same adapter under different VRAM/epoch
# budgets). Listed here as (family, output_name) pairs.
_INTENTIONAL_OUTPUT_NAME_COLLISIONS: set[tuple[str, str]] = {
    ("lora", "anima"),
    ("tlora", "anima_tlora_ortho"),
    ("hydralora", "anima_hydra"),
}


def _builtin_files() -> list[Path]:
    """Direct children of gui-methods/ (skipping the ``custom/`` subdir)."""
    return sorted(p for p in GUI_METHODS_DIR.glob("*.toml") if p.is_file())


BUILTINS = _builtin_files()
assert BUILTINS, "no built-in gui-methods/*.toml files found"


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_builtin_has_variant_metadata(path: Path):
    """Every built-in must declare ``[variant].family``."""
    data = toml.loads(path.read_text(encoding="utf-8"))
    meta = data.get("variant")
    assert isinstance(meta, dict), f"{path.name}: missing [variant] table"
    family = meta.get("family")
    assert isinstance(family, str) and family, (
        f"{path.name}: [variant].family must be a non-empty string"
    )


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_builtin_has_no_legacy_router_keys(path: Path):
    """Three-axis routing replaced ``use_hydra`` / ``use_sigma_router`` /
    ``use_fei_router``. Stale TOMLs must be cleaned out."""
    data = toml.loads(path.read_text(encoding="utf-8"))
    for key in _LEGACY_ROUTER_KEYS:
        assert key not in data, (
            f"{path.name}: legacy router key {key!r} — replace with the "
            "three-axis surface (use_moe_style / route_per_layer / router_source)"
        )


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_builtin_loads_clean(path: Path, caplog):
    """Every built-in must merge cleanly through the base → preset → method
    chain — no schema warnings, no missing files."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        merged = load_method_preset(
            path.stem, "default", methods_subdir="gui-methods"
        )
    # Required keys must show up downstream — guarantees the merge didn't
    # silently drop everything.
    assert "network_module" in merged
    assert "output_name" in merged
    offenders = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and rec.name.startswith("library.")
    ]
    assert not offenders, f"{path.name} produced warnings: {offenders}"


def test_no_unintentional_output_name_collisions():
    """Built-in variants must each pick a unique ``output_name`` unless the
    (family, name) pair is in ``_INTENTIONAL_OUTPUT_NAME_COLLISIONS`` — i.e.
    multiple files baking the same adapter on purpose.
    """
    by_output: dict[str, list[tuple[str, str]]] = {}
    for path in BUILTINS:
        data = toml.loads(path.read_text(encoding="utf-8"))
        output_name = data.get("output_name")
        family = (data.get("variant") or {}).get("family", "?")
        if not isinstance(output_name, str):
            continue
        by_output.setdefault(output_name, []).append((family, path.stem))

    bad: list[str] = []
    for output_name, owners in by_output.items():
        if len(owners) == 1:
            continue
        families = {f for f, _ in owners}
        if len(families) == 1 and (next(iter(families)), output_name) in _INTENTIONAL_OUTPUT_NAME_COLLISIONS:
            continue
        owner_str = ", ".join(f"{stem} (family={family})" for family, stem in owners)
        bad.append(f"output_name={output_name!r} shared by: {owner_str}")
    assert not bad, "Output name collisions:\n  " + "\n  ".join(bad)


def test_variant_discovery_matches_filesystem():
    """``list_gui_variants`` should surface every built-in whose ``[variant]
    .family`` matches the requested family — no hardcoded map drift."""
    from gui import list_gui_variants

    expected_by_family: dict[str, set[str]] = {}
    for path in BUILTINS:
        data = toml.loads(path.read_text(encoding="utf-8"))
        family = (data.get("variant") or {}).get("family")
        if isinstance(family, str) and family:
            expected_by_family.setdefault(family, set()).add(path.stem)

    for family, expected in expected_by_family.items():
        listed = {v for v in list_gui_variants(family) if not v.startswith("custom/")}
        assert listed == expected, (
            f"family={family}: list_gui_variants returned {sorted(listed)}, "
            f"expected {sorted(expected)}"
        )
