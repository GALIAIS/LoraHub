"""YAML config loader and JSON Schema exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lorahub.core.config.schema import TrainingConfig


def load_config(path: Path) -> TrainingConfig:
    """Load and validate a config YAML file."""
    raw = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw) or {}
    return TrainingConfig.model_validate(data)


def dump_config(config: TrainingConfig, path: Path) -> None:
    """Serialize a TrainingConfig back to YAML using camelCase aliases."""
    data = config.model_dump(mode="json", exclude_none=True, by_alias=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def export_json_schema(path: Path | None = None) -> str:
    """Export the config JSON Schema (for UI form generation)."""
    schema = TrainingConfig.model_json_schema()
    text = json.dumps(schema, indent=2, ensure_ascii=False)
    if path is not None:
        path.write_text(text, encoding="utf-8")
    return text
