"""Dataset scanning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from lorahub.api.helpers import _scan_dataset_path

router = APIRouter(prefix="/api")


@router.get("/datasets/scan")
def scan_dataset(path: str, recursive: bool = False, limit: int = 40) -> dict[str, Any]:
    return _scan_dataset_path(Path(path), recursive=recursive, limit=limit)
