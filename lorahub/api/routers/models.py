"""Model download endpoint.

POST /api/models/download starts a synchronous download. For small models or
quick metadata checks it returns when finished; the route is kept simple
since users typically run it once at setup time. Long-running downloads
should be parallelised by the client (one request per model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.core.models.downloader import DownloadRequest, download

router = APIRouter(prefix="/api")


class DownloadModelRequest(BaseModel):
    source: Literal["huggingface", "modelscope"]
    repo_id: str  # "owner/name"
    revision: str = "master"
    target_dir: str | None = None  # absolute or workspace-relative


@router.post("/models/download")
def download_model(req: DownloadModelRequest) -> dict[str, Any]:
    if not req.repo_id or "/" not in req.repo_id:
        raise HTTPException(status_code=400, detail="repo_id must be 'owner/name'")

    settings = app_module._settings_store.load()
    target = Path(req.target_dir).expanduser().resolve() if req.target_dir else None
    download_req = DownloadRequest(
        source=req.source,
        repo_id=req.repo_id,
        revision=req.revision,
        target_dir=target,
        huggingface_endpoint=settings.huggingface_endpoint,
        modelscope_token=settings.modelscope_token,
    )

    try:
        result = download(download_req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "source": req.source,
        "repo_id": req.repo_id,
        "revision": req.revision,
        "target": str(result.target),
        "files": result.files,
        "total_bytes": result.total_bytes,
    }
