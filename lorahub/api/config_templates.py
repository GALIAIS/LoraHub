"""Built-in config templates served by ``GET /api/configs/templates``.

The web UI lets the user spawn a new config from one of these starting points
instead of having to fill the whole form from scratch. Templates live as
plain YAML files under ``configs/`` (shipped with the repo) so they can be
edited without touching Python; each one is parsed and validated with
:class:`TrainingConfig` before being added to the catalogue. A bad template is
logged and skipped so a typo in one file can't take the whole endpoint down.

Path-bearing fields (``checkpoint`` / ``dataset.source``) are intentionally
left blank: ``pathlib.Path("")`` is a valid Path, so the schema accepts it,
and the user is expected to fill the real path in the form before saving.

Each template YAML may carry two optional top-level metadata blocks:

* ``_template``: ``{name, description, arch}`` — UI card metadata.
* ``_placeholders``: a list of ``{key, label, path_field, placeholder}`` entries
  describing the values the user must supply when instantiating the template
  (e.g. ``base_model.checkpoint``). The web UI renders these as a fill-in
  form before saving the config so users no longer hand-edit YAML.

Both blocks are stripped before validation because :class:`TrainingConfig` has
``extra="forbid"``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from lorahub.api.dataset_files import is_link_like
from lorahub.core.config.schema import TrainingConfig

logger = logging.getLogger(__name__)

# Default location, relative to the repo root. Resolved relative to this file
# rather than CWD so ``lorahub serve`` works from any directory.
_DEFAULT_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "configs" / "builtin"
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Fallback metadata used when a YAML file omits the ``_template`` block.
_FALLBACK_NAME_FROM_ID = {
    "anima_lora_default": "Anima LoRA Default",
    "anima_lora_8gb": "Anima LoRA 8GB",
    "anima_loha_32gb": "Anima LoHA 32GB",
    "anima_lokr_32gb": "Anima LoKr 32GB",
    "anima_lora_v100_fp16": "Anima LoRA V100 FP16",
    "sdxl_character": "SDXL Character",
    "sdxl_style": "SDXL Style",
    "sd15_character": "SD 1.5 Character",
    "blank": "Blank",
}

_FALLBACK_DESCRIPTION_FROM_ID = {
    "anima_lora_default": "Anima 通用基线；用于对照，不是画风强化配方。",
    "anima_lora_8gb": "Anima 8GB 安全档：768 分辨率、低显存优化。",
    "anima_loha_32gb": "32GB LoHA 配方：batchSize 2、gradAccum 4、10 epoch、CMMD 验证。",
    "anima_lokr_32gb": "32GB factorized LoKr 配方：rank 8、batchSize 1、gradAccum 8、checkpointing on、compile off。",
    "anima_lora_v100_fp16": "V100 兼容档：fp16、PyTorch SDPA、关闭 torch.compile。",
}


def _coerce_meta(stem: str, meta: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Build the {name, description, arch} dict the UI expects.

    Reads from the ``_template`` block when present; falls back to sensible
    defaults derived from the file stem and the config body so a YAML can
    omit the metadata entirely and still appear in the catalogue.
    """
    meta = meta if isinstance(meta, dict) else {}
    name = str(meta.get("name") or _FALLBACK_NAME_FROM_ID.get(stem, stem))
    description = str(
        meta.get("description") or _FALLBACK_DESCRIPTION_FROM_ID.get(stem, "")
    )
    arch = str(
        meta.get("arch")
        or config.get("base_model", {}).get("arch")
        or "sdxl"
    )
    return {"name": name, "description": description, "arch": arch}


def _coerce_placeholders(raw: Any) -> list[dict[str, str]]:
    """Normalise the optional ``_placeholders`` list.

    Each entry must have ``key``, ``label``, ``path_field`` and (optionally)
    ``placeholder``. Anything else is dropped silently — placeholders are
    purely a UI affordance, so a malformed entry should not break the
    template; it just won't render an extra input field.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        label = item.get("label")
        path_field = item.get("path_field")
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(path_field, str) or not path_field.strip():
            continue
        out.append(
            {
                "key": key.strip(),
                "label": str(label or key).strip(),
                "path_field": path_field.strip(),
                "placeholder": str(item.get("placeholder") or ""),
            }
        )
    return out


def _set_by_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Walk ``dotted`` (``a.b.c``) into ``target`` and set the leaf to ``value``.

    Intermediate mappings are created if they don't exist; non-mapping
    intermediates raise ``ValueError`` so we don't silently overwrite scalar
    data sitting where a sub-tree should be. This is the tiny dotted-path
    setter the placeholder system needs — no jsonpath dependency.
    """
    if not dotted:
        msg = "path_field must not be empty"
        raise ValueError(msg)
    parts = dotted.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            msg = f"path {dotted!r} traverses non-mapping at {part!r}"
            raise ValueError(msg)
        nxt = cursor.get(part)
        if nxt is None:
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    if not isinstance(cursor, dict):
        msg = f"path {dotted!r} traverses non-mapping leaf"
        raise ValueError(msg)
    cursor[parts[-1]] = value


def apply_placeholders(
    config: dict[str, Any],
    placeholders: list[dict[str, str]],
    values: dict[str, Any],
) -> dict[str, Any]:
    """Return a deep copy of ``config`` with placeholder values substituted.

    Only declared placeholder keys are honoured — extra keys in ``values``
    are ignored so the caller can reuse a single dict across templates.
    Empty / missing values leave the field untouched, which is fine because
    the schema accepts empty strings for path-shaped fields.
    """
    out = copy.deepcopy(config)
    for ph in placeholders:
        key = ph["key"]
        if key not in values:
            continue
        raw = values[key]
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        _set_by_path(out, ph["path_field"], text)
    return out


def _load_one(path: Path) -> dict[str, Any] | None:
    """Parse + validate a single template YAML.

    Returns the template dict (id/name/description/arch/config) on success,
    or ``None`` when the file can't be read or fails schema validation. All
    failure paths log a warning so a bad template is visible in server logs
    but doesn't abort startup.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("skipping config template %s: cannot read yaml: %s", path.name, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("skipping config template %s: top-level yaml must be a mapping", path.name)
        return None

    # Pop metadata before validation — TrainingConfig has extra="forbid".
    meta = data.pop("_template", None)
    placeholders_raw = data.pop("_placeholders", None)
    try:
        TrainingConfig.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("skipping config template %s: validation failed: %s", path.name, exc)
        return None

    info = _coerce_meta(path.stem, meta, data)
    placeholders = _coerce_placeholders(placeholders_raw)
    return {
        "id": path.stem,
        "name": info["name"],
        "description": info["description"],
        "arch": info["arch"],
        "placeholders": placeholders,
        "config": data,
    }


def load_templates(directory: Path | None = None) -> list[dict[str, Any]]:
    """Discover, validate and return every YAML template.

    By default this scans the current top-level ``configs/*.yaml`` templates
    plus the legacy ``configs/builtin/*.yaml`` templates. Results are sorted by
    id for stable ordering in the UI. Passing ``directory`` keeps the old test
    hook semantics and scans only that folder.
    """
    bases = [directory] if directory is not None else [_DEFAULT_CONFIG_DIR, _DEFAULT_BUILTIN_DIR]
    templates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in bases:
        if base is None:
            continue
        if not base.is_dir():
            logger.warning("config templates directory %s is missing", base)
            continue
        for path in sorted(base.glob("*.y*ml")):
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            if is_link_like(path):
                logger.warning("skipping linked config template %s", path)
                continue
            if path.stem in seen:
                continue
            tpl = _load_one(path)
            if tpl is not None:
                seen.add(path.stem)
                templates.append(tpl)
    return sorted(templates, key=lambda item: item["id"])


# Eager-load once at import time so a cold start surfaces obviously-broken
# YAML in the logs immediately rather than the first time someone calls the
# endpoint. The router exposes ``load_templates`` directly so tests can
# point it at a tmp directory without touching this cached list.
TEMPLATES: list[dict[str, Any]] = load_templates()


__all__ = ["TEMPLATES", "apply_placeholders", "load_templates"]
