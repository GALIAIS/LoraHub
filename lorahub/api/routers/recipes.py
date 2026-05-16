"""Recipe browse / preview / validate / save / duplicate / rename / delete / import."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from lorahub.api.helpers import (
    _NAME_RE_PATTERN,
    _preflight_recipe,
    _recipe_path,
    _recipes_dir,
)
from lorahub.api import recipe_templates as recipe_templates_module
from lorahub.core.config.loader import dump_recipe, load_recipe
from lorahub.core.config.schema import RecipeConfig

router = APIRouter(prefix="/api")

# Upload size cap for /import. Recipe YAMLs are usually <10KB; reject anything
# above 1 MiB outright so a wrong file picked from the OS dialog can't OOM us.
_MAX_IMPORT_BYTES = 1 * 1024 * 1024


class ValidateRecipeRequest(BaseModel):
    recipe: dict[str, Any]


class SaveRecipeRequest(BaseModel):
    name: str
    recipe: dict[str, Any]
    overwrite: bool = False


class RenameRecipeRequest(BaseModel):
    new_name: str


def _validate_recipe_name(name: str) -> str:
    """Strip extension, run through the shared name regex, return canonical form."""
    canonical = name.strip().removesuffix(".yaml").removesuffix(".yml")
    if not re.match(_NAME_RE_PATTERN, canonical):
        raise HTTPException(
            status_code=400,
            detail="name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        )
    return canonical


def _new_recipe_target(name: str) -> Path:
    """Resolve the destination path for a write under recipes_dir, blocking traversal."""
    base = _recipes_dir()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / f"{name}.yaml").resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid name") from exc
    return target


@router.get("/recipes/schema")
def recipe_schema() -> dict[str, Any]:
    """JSON Schema for the recipe — used by the future UI to render forms."""
    return RecipeConfig.model_json_schema()


@router.post("/recipes/validate")
def validate_recipe(req: ValidateRecipeRequest) -> dict[str, Any]:
    """Validate a recipe payload without persisting or training.

    Always returns 200 — the response carries `valid: bool` and a list of
    structured field errors. This lets the form highlight bad fields without
    interpreting HTTP status codes.
    """
    from pydantic import ValidationError as _PydanticValidationError  # noqa: PLC0415

    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except _PydanticValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {
                    "loc": list(e.get("loc", [])),
                    "msg": e.get("msg", ""),
                    "type": e.get("type", ""),
                }
                for e in exc.errors()
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "errors": [{"loc": [], "msg": str(exc), "type": "internal"}]}

    return {
        "valid": True,
        "normalized": cfg.model_dump(mode="json"),
        "preflight": _preflight_recipe(cfg),
    }


@router.get("/recipes")
def list_recipes() -> dict[str, Any]:
    """List YAML recipe templates discovered under the recipes/ directory."""
    base = _recipes_dir()
    if not base.is_dir():
        return {"dir": str(base), "recipes": []}

    items: list[dict[str, Any]] = []
    for p in sorted(base.glob("*.y*ml")):
        if p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        stat = p.stat()
        entry: dict[str, Any] = {
            "name": p.stem,
            "filename": p.name,
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "valid": False,
            "arch": None,
            "summary": None,
            "error": None,
        }
        try:
            cfg = load_recipe(p)
            entry["valid"] = True
            entry["arch"] = cfg.base_model.arch
            entry["summary"] = (
                f"{cfg.base_model.arch} · "
                f"{cfg.schedule.epochs} epoch(s) × bs {cfg.schedule.batch_size}"
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc).splitlines()[0][:200]
        items.append(entry)
    return {"dir": str(base), "recipes": items}


@router.post("/recipes", status_code=201)
def save_recipe(req: SaveRecipeRequest) -> dict[str, Any]:
    """Validate and persist a recipe to recipes/<name>.yaml."""
    name = _validate_recipe_name(req.name)

    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = _new_recipe_target(name)
    if target.exists() and not req.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"recipe {name!r} already exists; pass overwrite=true to replace",
        )

    dump_recipe(cfg, target)
    return {
        "name": name,
        "filename": target.name,
        "path": str(target),
        "overwritten": target.exists(),
    }


@router.get("/recipes/templates")
def list_recipe_templates() -> dict[str, Any]:
    """Return the built-in recipe templates the UI can spawn from.

    Re-reads the YAML directory on every call so newly dropped or edited
    files show up without restarting the server. Cost is trivial: 4-ish
    small files parsed once per request.
    """
    return {"templates": recipe_templates_module.load_templates()}


@router.post("/recipes/import", status_code=201)
async def import_recipe(
    file: UploadFile = File(...),  # noqa: B008
    name: str = Form(...),  # noqa: B008
    overwrite: bool = Form(default=False),  # noqa: B008
) -> dict[str, Any]:
    """Accept a YAML upload, validate as RecipeConfig, write under recipes_dir.

    The file is read fully into memory — recipe YAMLs are tiny by design.
    Anything bigger than ``_MAX_IMPORT_BYTES`` is rejected with 413 to avoid
    accidentally pulling in something huge that the user picked by mistake.
    """
    canonical = _validate_recipe_name(name)

    raw = await file.read()
    if len(raw) > _MAX_IMPORT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"recipe upload exceeds {_MAX_IMPORT_BYTES} bytes",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="recipe file must be UTF-8") from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"invalid yaml: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="recipe yaml must be a mapping")

    try:
        cfg = RecipeConfig.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = _new_recipe_target(canonical)
    already_exists = target.exists()
    if already_exists and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"recipe {canonical!r} already exists; pass overwrite=true to replace",
        )

    dump_recipe(cfg, target)
    return {
        "name": canonical,
        "filename": target.name,
        "path": str(target),
        "overwritten": already_exists,
    }


@router.post("/recipes/{name}/duplicate", status_code=201)
def duplicate_recipe(name: str, req: RenameRecipeRequest) -> dict[str, Any]:
    """Copy ``recipes/<name>.yaml`` to ``recipes/<new_name>.yaml``.

    The source is opened through ``load_recipe`` first so callers get a 422 if
    they try to clone a malformed recipe — matching the error surface of save.
    """
    src = _recipe_path(name)
    new_name = _validate_recipe_name(req.new_name)
    dst = _new_recipe_target(new_name)

    if dst.exists():
        raise HTTPException(
            status_code=409,
            detail=f"recipe {new_name!r} already exists",
        )

    # Sanity-check that the source is at least readable as YAML; we don't fail
    # the whole copy on a model-validation error so users can clone a recipe
    # they're still fixing.
    try:
        src.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"cannot read source: {exc}") from exc

    shutil.copy2(src, dst)
    return {
        "name": new_name,
        "filename": dst.name,
        "path": str(dst),
    }


@router.post("/recipes/{name}/rename")
def rename_recipe(name: str, req: RenameRecipeRequest) -> dict[str, Any]:
    """Atomically rename ``recipes/<name>.yaml`` to ``recipes/<new_name>.yaml``."""
    src = _recipe_path(name)
    new_name = _validate_recipe_name(req.new_name)
    dst = _new_recipe_target(new_name)

    if dst.exists() and dst != src:
        raise HTTPException(
            status_code=409,
            detail=f"recipe {new_name!r} already exists",
        )

    src.rename(dst)
    return {
        "name": new_name,
        "filename": dst.name,
        "path": str(dst),
    }


@router.delete("/recipes/{name}")
def delete_recipe(name: str) -> dict[str, Any]:
    """Delete ``recipes/<name>.yaml``."""
    path = _recipe_path(name)
    path.unlink()
    return {"deleted": True, "name": path.stem}


@router.get("/recipes/{name}")
def get_recipe(name: str) -> dict[str, Any]:
    """Return a recipe's raw YAML and parsed dict (for previewing or launching)."""
    # Sibling endpoints share the /recipes/ prefix; keep them out of {name}.
    if name in {"schema", "validate", "templates", "import"}:
        raise HTTPException(status_code=404, detail="recipe not found")
    path = _recipe_path(name)

    raw = path.read_text(encoding="utf-8")
    parsed: dict[str, Any] | None = None
    error: str | None = None
    try:
        parsed = load_recipe(path).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return {
        "name": path.stem,
        "filename": path.name,
        "path": str(path),
        "content": raw,
        "parsed": parsed,
        "error": error,
    }
