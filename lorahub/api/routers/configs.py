"""Config browse / preview / validate / save / duplicate / rename / delete / import."""

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
    _preflight_config,
    _config_path,
    _configs_dir,
)
from lorahub.api import config_templates as config_templates_module
from lorahub.core.config.loader import dump_config, load_config
from lorahub.core.config.schema import TrainingConfig

router = APIRouter(prefix="/api")

# Upload size cap for /import. config yamls are usually <10KB; reject anything
# above 1 MiB outright so a wrong file picked from the OS dialog can't OOM us.
_MAX_IMPORT_BYTES = 1 * 1024 * 1024


# Heuristics for content the user likely pasted by accident. We surface a
# targeted hint when the file looks like one of these instead of dumping
# the raw PyYAML scanner error — the scanner message ("mapping values are
# not allowed here") tells you nothing about *why* the file is wrong if
# you didn't write it yourself.
_NON_YAML_SIGNATURES: tuple[tuple[str, str, str], ...] = (
    # (regex pattern, kind, friendly hint shown to the user)
    (
        r"^\s*<!doctype\s+html|^\s*<html|^\s*<head\b",
        "html",
        "文件看起来是 HTML 页面源码,而不是 YAML 配置。常见原因:从浏览器 "
        "「另存为」保存了网页本体而不是 raw 文本,或复制粘贴时连同页面代码"
        "一起拷贝了。请到 GitHub 上点 Raw 按钮后再保存,或确认文件首行是 "
        "schemaVersion / baseModel 等 YAML 字段。",
    ),
    (
        r"--[A-Za-z][\w-]*\s*:.*[,;]\s*$|^\s*:root\s*\{|^\s*\.[\w-]+\s*\{",
        "css",
        "文件中混入了 CSS 代码片段(如 `--fontStack-monospace: \"...\"` 或 "
        "`:root { ... }`),YAML 解析器无法处理。这通常是从某个网页直接拷贝 "
        "时连同样式表一起复制了。请重新打开**原始 YAML 文件**(通常在 "
        "`configs/` 目录下,后缀 `.yaml`)再上传。",
    ),
    (
        r"^\s*\{[\s\S]*\"[\w-]+\"\s*:\s*",
        "json",
        "文件内容是 JSON 而不是 YAML。LoraHub 的导入端点只接受 YAML(.yaml / .yml)。"
        "如需从 JSON 转换,可在本地用 `python -c \"import json,yaml;"
        "yaml.safe_dump(json.load(open('x.json')))\"` 转换后再上传。",
    ),
    (
        r"^\s*#\s*!.*python|^\s*import\s+\w+|^\s*from\s+\w+\s+import\s",
        "python",
        "文件像是 Python 脚本(包含 import / from 等关键字)。你可能误选了 "
        "scripts/ 目录下的训练脚本,而不是 configs/ 目录下的 YAML 配置。",
    ),
)


def _detect_non_yaml_kind(text: str) -> tuple[str, str] | None:
    """Identify common 'wrong file type' patterns.

    Returns ``(kind, hint)`` if the head of the file matches one of the
    known accidental-paste signatures, or ``None`` if nothing matches.
    Only the first 4 KiB of the file is scanned — that's enough to spot
    HTML doctypes / CSS rules / JSON braces while staying fast for the
    legitimate-but-malformed case where the rest of the file might be
    real YAML.
    """
    head = text[:4096]
    for pattern, kind, hint in _NON_YAML_SIGNATURES:
        if re.search(pattern, head, re.IGNORECASE | re.MULTILINE):
            return kind, hint
    return None


def _yaml_error_detail(exc: yaml.YAMLError, text: str) -> dict[str, Any]:
    """Build a structured error payload the frontend can render nicely.

    Always includes the raw PyYAML message + line / column when present.
    Adds a friendly ``hint`` field when the file looks like HTML / CSS /
    JSON / Python — typical accidental-paste cases that a bare scanner
    error doesn't hint at.
    """
    payload: dict[str, Any] = {
        "type": "yaml_parse_error",
        "message": str(exc),
    }
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        # PyYAML uses 0-based line/column; humans expect 1-based.
        payload["line"] = mark.line + 1
        payload["column"] = mark.column + 1
        # Pull the offending line so the frontend can render it inline
        # without the user having to open the file.
        try:
            offending = text.splitlines()[mark.line]
            # Clip absurdly long lines (minified HTML etc).
            if len(offending) > 200:
                offending = offending[:200] + "…"
            payload["snippet"] = offending
        except IndexError:
            pass

    detected = _detect_non_yaml_kind(text)
    if detected is not None:
        kind, hint = detected
        payload["kind"] = kind
        payload["hint"] = hint
    return payload


class ValidateConfigRequest(BaseModel):
    config: dict[str, Any]


class SaveConfigRequest(BaseModel):
    name: str
    config: dict[str, Any]
    overwrite: bool = False


class RenameConfigRequest(BaseModel):
    new_name: str


class InstantiateTemplateRequest(BaseModel):
    name: str
    values: dict[str, Any] = {}
    overwrite: bool = False


def _validate_config_name(name: str) -> str:
    """Strip extension, run through the shared name regex, return canonical form."""
    canonical = name.strip().removesuffix(".yaml").removesuffix(".yml")
    if not re.match(_NAME_RE_PATTERN, canonical):
        raise HTTPException(
            status_code=400,
            detail="name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        )
    return canonical


def _new_config_target(name: str) -> Path:
    """Resolve the destination path for a write under configs_dir, blocking traversal."""
    base = _configs_dir()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / f"{name}.yaml").resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid name") from exc
    return target


@router.get("/configs/schema")
def config_schema() -> dict[str, Any]:
    """JSON Schema for the config — used by the future UI to render forms."""
    return TrainingConfig.model_json_schema()


@router.post("/configs/validate")
def validate_config(req: ValidateConfigRequest) -> dict[str, Any]:
    """Validate a config payload without persisting or training.

    Always returns 200 — the response carries `valid: bool` and a list of
    structured field errors. This lets the form highlight bad fields without
    interpreting HTTP status codes.
    """
    from pydantic import ValidationError as _PydanticValidationError  # noqa: PLC0415

    try:
        cfg = TrainingConfig.model_validate(req.config)
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
        "normalized": cfg.model_dump(mode="json", by_alias=True),
        "preflight": _preflight_config(cfg),
    }


# ----------------------------------------------------------------------- #
# /configs/llm-advise — LLM-driven recommendation
#
# The user clicks "智能推荐", types their intent in a textarea (e.g.
# "character LoRA, ~4h train budget"), and the backend hands the
# already-configured AI provider a strict-format prompt that includes
# hardware budget + dataset stats + the schema field catalogue. The
# LLM returns a full config + per-field patches; the route validates
# the proposal and returns it to the UI so the user can accept all /
# accept-some / discard.
# ----------------------------------------------------------------------- #


class LlmAdviseRequest(BaseModel):
    """User-facing request body for ``/api/configs/llm-advise``.

    ``currentCfg`` mirrors the form state (camelCase, before
    schema-validation) so the LLM can observe the user's starting
    point. ``intent`` is a free-form text the user types in the UI's
    textarea — the prompt template inlines it verbatim.

    Hardware / dataset fields are optional: when omitted the route
    probes nvidia-smi + ``count_images`` itself so the LLM still
    sees real data without forcing the UI to plumb everything
    through.
    """

    currentCfg: dict[str, Any]
    intent: str = ""
    vramMib: int | None = None
    gpuName: str | None = None
    datasetPath: str | None = None
    datasetImageCount: int | None = None


class LlmAdviseResponse(BaseModel):
    rationale: str
    patches: list[dict[str, Any]]
    fullConfig: dict[str, Any]
    validationIssues: list[dict[str, Any]]
    providerId: str
    modelId: str
    elapsedMs: int


@router.post("/configs/llm-advise", response_model=LlmAdviseResponse)
def llm_advise_config(req: LlmAdviseRequest) -> LlmAdviseResponse:
    """LLM-driven config recommendation — **temporarily disabled**.

    The advisor was wired through ``lorahub.api.config_advisor_llm.
    run_advisor`` but the upstream LLM proxy fronting our configured
    AI provider drops multi-thousand-token prompts at the 60 s mark
    with ``Server disconnected without sending a response``. Until
    that's reworked onto a streaming code path, the route shorts out
    with a 503 and refuses to invoke the upstream — that way no
    confused user clicks the (now hidden) UI button and waits a full
    minute to see a 422.

    The implementation lives in ``config_advisor_llm.py`` and is left
    intact so a one-line revert restores it once the upstream issue
    is fixed.
    """
    raise HTTPException(
        status_code=503,
        detail=(
            "智能推荐已暂时停用 (上游 LLM 流量层在长 prompt 上 60s 断流问题)。"
            "代码与路由仍在,等切换到 streaming 路径后会恢复。"
        ),
    )


@router.get("/configs")
def list_configs() -> dict[str, Any]:
    """List YAML config templates discovered under the configs/ directory."""
    base = _configs_dir()
    if not base.is_dir():
        return {"dir": str(base), "configs": []}

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
            "backend": None,
            "summary": None,
            "error": None,
        }
        try:
            cfg = load_config(p)
            entry["valid"] = True
            entry["arch"] = cfg.base_model.arch
            entry["backend"] = cfg.backend.type
            entry["summary"] = (
                f"{cfg.base_model.arch} · "
                f"{cfg.schedule.epochs} epoch(s) × bs {cfg.schedule.batch_size}"
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc).splitlines()[0][:200]
        items.append(entry)
    return {"dir": str(base), "configs": items}


@router.post("/configs", status_code=201)
def save_config(req: SaveConfigRequest) -> dict[str, Any]:
    """Validate and persist a config to configs/<name>.yaml."""
    name = _validate_config_name(req.name)

    try:
        cfg = TrainingConfig.model_validate(req.config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = _new_config_target(name)
    if target.exists() and not req.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"config {name!r} already exists; pass overwrite=true to replace",
        )

    dump_config(cfg, target)
    return {
        "name": name,
        "filename": target.name,
        "path": str(target),
        "overwritten": target.exists(),
    }


@router.get("/configs/templates")
def list_config_templates() -> dict[str, Any]:
    """Return the built-in config templates the UI can spawn from.

    Re-reads the YAML directory on every call so newly dropped or edited
    files show up without restarting the server. Cost is trivial: 4-ish
    small files parsed once per request.
    """
    return {"templates": config_templates_module.load_templates()}


@router.post("/configs/templates/{template_id}/instantiate", status_code=201)
def instantiate_config_template(
    template_id: str,
    req: InstantiateTemplateRequest,
) -> dict[str, Any]:
    """Materialise a template into a fresh config at ``configs/<name>.yaml``.

    Steps:
      1. Look up the template by id (404 if absent).
      2. Apply the user-supplied placeholder values onto a deep copy of the
         template body via dotted-path setters.
      3. Validate the result through ``TrainingConfig`` (422 on failure).
      4. Persist using the same name validation / overwrite guard the regular
         save endpoint uses (400 / 409 on conflicts).
    """
    name = _validate_config_name(req.name)

    templates = config_templates_module.load_templates()
    template = next((t for t in templates if t["id"] == template_id), None)
    if template is None:
        raise HTTPException(status_code=404, detail="template not found")

    try:
        body = config_templates_module.apply_placeholders(
            template["config"], template.get("placeholders", []), req.values
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        cfg = TrainingConfig.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = _new_config_target(name)
    if target.exists() and not req.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"config {name!r} already exists; pass overwrite=true to replace",
        )

    dump_config(cfg, target)
    return {
        "name": name,
        "filename": target.name,
        "path": str(target),
        "template_id": template_id,
    }


@router.post("/configs/import", status_code=201)
async def import_config(
    file: UploadFile = File(...),  # noqa: B008
    name: str = Form(...),  # noqa: B008
    overwrite: bool = Form(default=False),  # noqa: B008
) -> dict[str, Any]:
    """Accept a YAML upload, validate as TrainingConfig, write under configs_dir.

    The file is read fully into memory — config yamls are tiny by design.
    Anything bigger than ``_MAX_IMPORT_BYTES`` is rejected with 413 to avoid
    accidentally pulling in something huge that the user picked by mistake.
    """
    canonical = _validate_config_name(name)

    raw = await file.read()
    if len(raw) > _MAX_IMPORT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"config upload exceeds {_MAX_IMPORT_BYTES} bytes",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "type": "encoding_error",
                "message": "config file must be UTF-8",
                "hint": (
                    "文件不是 UTF-8 编码。Windows 记事本另存为时请选择 "
                    "「UTF-8」(不要选「UTF-8 with BOM」或 ANSI)。"
                ),
            },
        ) from exc

    # Pre-flight check before YAML parsing: if the file looks like an
    # entirely different format we can give a much more specific
    # diagnostic than PyYAML's scanner ever will.
    detected = _detect_non_yaml_kind(text)
    if detected is not None:
        kind, hint = detected
        raise HTTPException(
            status_code=422,
            detail={
                "type": "wrong_file_type",
                "kind": kind,
                "message": f"file does not look like YAML (detected: {kind})",
                "hint": hint,
            },
        )

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422,
            detail=_yaml_error_detail(exc, text),
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "type": "not_a_mapping",
                "message": "config yaml must be a mapping",
                "hint": (
                    "YAML 文件根节点必须是 key: value 形式的映射。"
                    "如果你看到的是一个列表(以 - 开头)或纯标量字符串,"
                    "说明文件格式不对。"
                ),
            },
        )

    try:
        cfg = TrainingConfig.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = _new_config_target(canonical)
    already_exists = target.exists()
    if already_exists and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"config {canonical!r} already exists; pass overwrite=true to replace",
        )

    dump_config(cfg, target)
    return {
        "name": canonical,
        "filename": target.name,
        "path": str(target),
        "overwritten": already_exists,
    }


@router.post("/configs/{name}/duplicate", status_code=201)
def duplicate_config(name: str, req: RenameConfigRequest) -> dict[str, Any]:
    """Copy ``configs/<name>.yaml`` to ``configs/<new_name>.yaml``.

    The source is opened through ``load_config`` first so callers get a 422 if
    they try to clone a malformed config — matching the error surface of save.
    """
    src = _config_path(name)
    new_name = _validate_config_name(req.new_name)
    dst = _new_config_target(new_name)

    if dst.exists():
        raise HTTPException(
            status_code=409,
            detail=f"config {new_name!r} already exists",
        )

    # Sanity-check that the source is at least readable as YAML; we don't fail
    # the whole copy on a model-validation error so users can clone a config
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


@router.post("/configs/{name}/rename")
def rename_config(name: str, req: RenameConfigRequest) -> dict[str, Any]:
    """Atomically rename ``configs/<name>.yaml`` to ``configs/<new_name>.yaml``."""
    src = _config_path(name)
    new_name = _validate_config_name(req.new_name)
    dst = _new_config_target(new_name)

    if dst.exists() and dst != src:
        raise HTTPException(
            status_code=409,
            detail=f"config {new_name!r} already exists",
        )

    src.rename(dst)
    return {
        "name": new_name,
        "filename": dst.name,
        "path": str(dst),
    }


@router.delete("/configs/{name}")
def delete_config(name: str) -> dict[str, Any]:
    """Delete ``configs/<name>.yaml``."""
    path = _config_path(name)
    path.unlink()
    return {"deleted": True, "name": path.stem}


@router.get("/configs/{name}")
def get_config(name: str) -> dict[str, Any]:
    """Return a config's raw YAML and parsed dict (for previewing or launching)."""
    # Sibling endpoints share the /configs/ prefix; keep them out of {name}.
    if name in {"schema", "validate", "templates", "import"}:
        raise HTTPException(status_code=404, detail="config not found")
    path = _config_path(name)

    raw = path.read_text(encoding="utf-8")
    parsed: dict[str, Any] | None = None
    error: str | None = None
    try:
        parsed = load_config(path).model_dump(mode="json", by_alias=True)
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
