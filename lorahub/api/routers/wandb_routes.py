"""Read-only proxy to the Weights & Biases public API.

The wandb run dashboard cannot be embedded as an iframe (wandb sends
``X-Frame-Options: DENY`` / CSP ``frame-ancestors 'none'``), so the
"训练分析 → W&B" tab fetches the metrics through these endpoints and
re-renders them with the project's own chart components.

Design notes
------------

* ``import wandb`` is **lazy**. The dev environment does not require
  wandb to be installed; only users who actually opt into wandb
  monitoring need the package. Endpoints catch ``ImportError`` and
  return a structured error so the UI can show a clear setup hint.
* The wandb run identity (``entity/project/run_id``) is recovered from
  the ``wandb_run_url`` we stamp on ``JobRecord.metadata`` in
  ``lorahub/api/jobs_helpers/lifecycle.py:_capture_wandb_run_url``.
  No new state is written.
* ``WANDB_API_KEY`` is read from settings (the same value that the
  job runner injects into training subprocesses); ``wandb.Api`` reads
  it from the env, but we pass it explicitly so the API server does
  not need to be restarted after a key change.
* ``base_url`` (self-hosted W&B Server) flows through
  ``settings.wandb_base_url`` once that setting is added; until then
  we fall back to env / ``Settings`` defaults.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api import state

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wandb")


# wandb URLs come in two shapes:
#   SaaS:        https://wandb.ai/<entity>/<project>/runs/<id>[?...]
#   self-hosted: https://wb.example.com/<entity>/<project>/runs/<id>
# Both are captured by the same regex.
_WANDB_RUN_PATH_RE = re.compile(
    r"^/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)


def _parse_run_url(url: str) -> tuple[str, str, str]:
    """Split ``https://host/entity/project/runs/run_id`` into its parts.

    Raises ``HTTPException(502)`` instead of bare ``ValueError`` so a
    malformed URL stamped on the job (e.g. user pasted the wandb sweep
    URL instead of a run URL) surfaces as a structured 502, not the
    framework's generic 500.
    """
    parsed = urlparse(url)
    match = _WANDB_RUN_PATH_RE.match(parsed.path or "")
    if match is None:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "wandb_run_url_invalid",
                "message": f"unrecognized wandb run url shape: {url!r}",
            },
        )
    return match.group("entity"), match.group("project"), match.group("run_id")


def _resolve_run_url(job_id: str) -> str:
    rec = state.registry.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    metadata = rec.metadata if isinstance(rec.metadata, dict) else {}
    url = metadata.get("wandb_run_url")
    if not isinstance(url, str) or not url:
        raise HTTPException(
            status_code=409,
            detail=(
                "job has no wandb run url stamped — make sure "
                "monitoring.enable_wandb=true and the run reached "
                "wandb.init() before failing."
            ),
        )
    return url


def _wandb_api():  # noqa: ANN202 — wandb types only available on demand
    """Construct ``wandb.Api`` lazily.

    Raises ``HTTPException`` with structured ``code`` so the UI can
    distinguish "wandb not installed" from "no api key" from generic
    network failures.
    """
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "wandb_not_installed",
                "message": "wandb package is not installed in the API environment",
            },
        ) from exc

    settings = app_module._settings_store.load()
    api_key = settings.wandb_api_key or None
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "wandb_api_key_missing",
                "message": "set wandb api key in 设置 → 网络 first",
            },
        )

    overrides: dict[str, str] = {}
    base_url = getattr(settings, "wandb_base_url", None)
    if base_url:
        overrides["base_url"] = base_url
    try:
        return wandb.Api(api_key=api_key, overrides=overrides, timeout=20)
    except Exception as exc:  # noqa: BLE001 — wandb raises CommError, AuthenticationError, etc.
        # Wrap as 503 so the UI can show the underlying error rather
        # than the framework's "Internal Server Error" placeholder.
        log.exception("wandb.Api construction failed")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "wandb_api_init_failed",
                "message": (
                    f"{type(exc).__name__}: {exc}. "
                    "检查 设置 → 网络 中的 W&B api key / base_url 是否正确,"
                    "以及当前服务可访问 wandb 后端。"
                ),
            },
        ) from exc


# --------------------------------------------------------- response models ---


class WandbStatusResponse(BaseModel):
    installed: bool
    api_key_configured: bool
    base_url: str | None = None


class WandbRunSummary(BaseModel):
    entity: str
    project: str
    run_id: str
    name: str | None
    state: str | None
    url: str
    config: dict[str, Any]
    summary: dict[str, Any]
    tags: list[str]


class WandbHistoryResponse(BaseModel):
    keys: list[str]
    rows: list[dict[str, Any]]
    sampled: bool
    samples_requested: int


# ------------------------------------------------------------- endpoints ----


@router.get("/status", response_model=WandbStatusResponse)
def wandb_status() -> WandbStatusResponse:
    """Probe whether the W&B integration is usable on this server."""
    settings = app_module._settings_store.load()
    try:
        import wandb  # noqa: PLC0415, F401

        installed = True
    except ImportError:
        installed = False
    return WandbStatusResponse(
        installed=installed,
        api_key_configured=bool(settings.wandb_api_key),
        base_url=getattr(settings, "wandb_base_url", None),
    )


@router.get("/runs/{job_id}/summary", response_model=WandbRunSummary)
def wandb_run_summary(job_id: str) -> WandbRunSummary:
    """Return run config + summary metrics + state for one job."""
    url = _resolve_run_url(job_id)
    entity, project, run_id = _parse_run_url(url)
    api = _wandb_api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:  # noqa: BLE001 — wandb raises bare CommError etc.
        raise HTTPException(
            status_code=502,
            detail={"code": "wandb_fetch_failed", "message": str(exc)},
        ) from exc

    try:
        summary_dict: dict[str, Any] = {}
        try:
            # ``run.summary`` is a SummaryDict whose nested values are
            # ``SummarySubDict`` instances (dict *subclasses*); pydantic v2
            # won't serialize a subclass through dict[str, Any], so we
            # recursively flatten to plain primitives here.
            raw = {k: v for k, v in dict(run.summary).items() if not k.startswith("_")}
            summary_dict = _coerce_jsonable(raw)
        except Exception:  # noqa: BLE001
            log.exception("wandb summary unwrap failed for %s", url)

        config_dict = _coerce_jsonable(dict(run.config)) if run.config else {}

        return WandbRunSummary(
            entity=entity,
            project=project,
            run_id=run_id,
            name=getattr(run, "name", None),
            state=getattr(run, "state", None),
            url=getattr(run, "url", url),
            config=config_dict,
            summary=summary_dict,
            tags=list(run.tags) if getattr(run, "tags", None) else [],
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Catch-all so a wandb-side schema change or unexpected attribute
        # access surfaces as a structured 502, not the framework's bare 500.
        log.exception("wandb run summary marshalling failed for %s", url)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "wandb_summary_marshal_failed",
                "message": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


@router.get("/runs/{job_id}/history", response_model=WandbHistoryResponse)
def wandb_run_history(
    job_id: str,
    keys: list[str] | None = Query(default=None),
    samples: int = Query(default=500, ge=1, le=10_000),
) -> WandbHistoryResponse:
    """Return sampled metric history for the given job's wandb run.

    Always uses ``run.history(samples=...)`` (the sampled/fast path)
    — ``scan_history`` is unbounded and would block the API server
    on a long run. The frontend is fine with sampling because it
    renders charts at screen resolution anyway.
    """
    url = _resolve_run_url(job_id)
    entity, project, run_id = _parse_run_url(url)
    api = _wandb_api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        df = run.history(samples=samples, keys=keys, pandas=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={"code": "wandb_fetch_failed", "message": str(exc)},
        ) from exc

    try:
        rows: list[dict[str, Any]]
        if df is None or len(df) == 0:
            rows = []
            out_keys: list[str] = []
        else:
            # Coerce numpy types / NaN / wandb subdicts to plain JSON
            # primitives so pydantic's default encoder is enough. Without
            # this, any image-file / histogram / SummarySubDict row would
            # blow up at response serialization.
            out_keys = [c for c in df.columns.tolist()]
            rows = [_coerce_jsonable(row) for row in df.to_dict(orient="records")]
        return WandbHistoryResponse(
            keys=out_keys,
            rows=rows,
            sampled=True,
            samples_requested=samples,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("wandb run history marshalling failed for %s", url)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "wandb_history_marshal_failed",
                "message": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


def _coerce_jsonable(obj: Any) -> Any:
    """Recursively rewrite wandb-side objects into plain JSON primitives.

    ``run.summary`` returns ``wandb.old.summary.SummarySubDict`` for nested
    fields (e.g. images, histograms). It's **not** a ``dict`` subclass —
    just a dict-like that exposes ``.items()`` / ``__getitem__`` — and its
    ``__getattr__`` raises ``KeyError`` instead of ``AttributeError`` on
    unknown names, which means ``getattr(x, "items", None)`` does *not*
    fall through to the default. Use explicit try/except around attribute
    access here so a dict-like wandb value doesn't blow up the coercion.

    Without this pass pydantic v2's serializer hits
    ``PydanticSerializationError: Unable to serialize unknown type`` on
    any nested wandb subdict and the endpoint surfaces as a bare 500.
    The same trap applies to numpy scalars sneaking out of
    ``run.history`` and to NaN floats — we normalise all three.
    """
    # Fast-path primitives so we don't spend a method-lookup on every leaf.
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if obj != obj else obj  # NaN → null
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Dict-like: builtin dict, OrderedDict, wandb's SummarySubDict, etc.
    # Use try/except instead of hasattr/getattr to dodge wandb's
    # KeyError-on-missing-attr __getattr__.
    try:
        items_fn = obj.items
    except (AttributeError, KeyError):
        items_fn = None
    if callable(items_fn):
        try:
            return {str(k): _coerce_jsonable(v) for k, v in items_fn()}
        except (TypeError, KeyError, AttributeError):
            pass  # not actually iterable as items — fall through

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_coerce_jsonable(v) for v in obj]

    # numpy scalars / pandas types expose .item() to unwrap to Python.
    try:
        item_fn = obj.item
    except (AttributeError, KeyError):
        item_fn = None
    if callable(item_fn):
        try:
            return _coerce_jsonable(item_fn())
        except Exception:  # noqa: BLE001
            pass

    return obj


__all__ = ["router"]
