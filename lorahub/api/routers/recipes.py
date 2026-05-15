"""Recipe browse / preview / validate / save."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api.helpers import (
    _NAME_RE_PATTERN,
    _preflight_recipe,
    _recipe_path,
    _recipes_dir,
)
from lorahub.core.config.loader import dump_recipe
from lorahub.core.config.schema import RecipeConfig

router = APIRouter(prefix="/api")


class ValidateRecipeRequest(BaseModel):
    recipe: dict[str, Any]


class SaveRecipeRequest(BaseModel):
    name: str
    recipe: dict[str, Any]
    overwrite: bool = False


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

    from lorahub.core.config.loader import load_recipe  # noqa: PLC0415

    items: list[dict[str, Any]] = []
    for p in sorted(base.glob("*.y*ml")):
        if p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        entry: dict[str, Any] = {
            "name": p.stem,
            "filename": p.name,
            "size": p.stat().st_size,
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
    import re  # noqa: PLC0415

    name = req.name.strip().removesuffix(".yaml").removesuffix(".yml")
    if not re.match(_NAME_RE_PATTERN, name):
        raise HTTPException(
            status_code=400,
            detail="name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        )

    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    base = _recipes_dir()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / f"{name}.yaml").resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid name") from exc

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


@router.get("/recipes/{name}")
def get_recipe(name: str) -> dict[str, Any]:
    """Return a recipe's raw YAML and parsed dict (for previewing or launching)."""
    if name in {"schema", "validate"}:  # sibling endpoints share the prefix
        raise HTTPException(status_code=404, detail="recipe not found")
    path = _recipe_path(name)

    from lorahub.core.config.loader import load_recipe  # noqa: PLC0415

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
