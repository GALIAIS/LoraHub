"""Built-in recipe templates served by ``GET /api/recipes/templates``.

The web UI lets the user spawn a new recipe from one of these starting points
instead of having to fill the whole form from scratch. Templates live as
plain YAML files under ``recipes/builtin/`` (shipped with the repo) so they
can be edited without touching Python; each one is parsed and validated with
:class:`RecipeConfig` before being added to the catalogue. A bad template is
logged and skipped so a typo in one file can't take the whole endpoint down.

Path-bearing fields (``checkpoint`` / ``dataset.source``) are intentionally
left blank: ``pathlib.Path("")`` is a valid Path, so the schema accepts it,
and the user is expected to fill the real path in the form before saving.

Each template YAML may carry an optional ``_template`` mapping at the top
level with ``name``, ``description``, and ``arch`` — these drive the UI card
and are stripped before validation (RecipeConfig has ``extra="forbid"``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from lorahub.core.config.schema import RecipeConfig

logger = logging.getLogger(__name__)

# Default location, relative to the repo root. The folder is shipped as part
# of the source tree (recipes/builtin/) and resolved relative to this file
# rather than CWD so ``lorahub serve`` works from any directory.
_DEFAULT_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "recipes" / "builtin"

# Fallback metadata used when a YAML file omits the ``_template`` block.
_FALLBACK_NAME_FROM_ID = {
    "sdxl_character": "SDXL Character",
    "sdxl_style": "SDXL Style",
    "sd15_character": "SD 1.5 Character",
    "blank": "Blank",
}


def _coerce_meta(stem: str, meta: Any, recipe: dict[str, Any]) -> dict[str, Any]:
    """Build the {name, description, arch} dict the UI expects.

    Reads from the ``_template`` block when present; falls back to sensible
    defaults derived from the file stem and the recipe body so a YAML can
    omit the metadata entirely and still appear in the catalogue.
    """
    meta = meta if isinstance(meta, dict) else {}
    name = str(meta.get("name") or _FALLBACK_NAME_FROM_ID.get(stem, stem))
    description = str(meta.get("description") or "")
    arch = str(
        meta.get("arch")
        or recipe.get("base_model", {}).get("arch")
        or "sdxl"
    )
    return {"name": name, "description": description, "arch": arch}


def _load_one(path: Path) -> dict[str, Any] | None:
    """Parse + validate a single template YAML.

    Returns the template dict (id/name/description/arch/recipe) on success,
    or ``None`` when the file can't be read or fails schema validation. All
    failure paths log a warning so a bad template is visible in server logs
    but doesn't abort startup.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("skipping recipe template %s: cannot read yaml: %s", path.name, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("skipping recipe template %s: top-level yaml must be a mapping", path.name)
        return None

    # Pop metadata before validation — RecipeConfig has extra="forbid".
    meta = data.pop("_template", None)
    try:
        RecipeConfig.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("skipping recipe template %s: validation failed: %s", path.name, exc)
        return None

    info = _coerce_meta(path.stem, meta, data)
    return {
        "id": path.stem,
        "name": info["name"],
        "description": info["description"],
        "arch": info["arch"],
        "recipe": data,
    }


def load_templates(directory: Path | None = None) -> list[dict[str, Any]]:
    """Discover, validate and return every YAML template under ``directory``.

    Results are sorted by id for stable ordering in the UI. The directory is
    re-scanned on every call so tests can swap it for a tmp_path without
    needing to reload modules.
    """
    base = directory if directory is not None else _DEFAULT_BUILTIN_DIR
    if not base.is_dir():
        logger.warning("recipe templates directory %s is missing", base)
        return []

    templates: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.y*ml")):
        if path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        tpl = _load_one(path)
        if tpl is not None:
            templates.append(tpl)
    return templates


# Eager-load once at import time so a cold start surfaces obviously-broken
# YAML in the logs immediately rather than the first time someone calls the
# endpoint. The router exposes ``load_templates`` directly so tests can
# point it at a tmp directory without touching this cached list.
TEMPLATES: list[dict[str, Any]] = load_templates()


__all__ = ["TEMPLATES", "load_templates"]
