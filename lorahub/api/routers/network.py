"""Network probe — measure latency to mirror candidates and pick the fastest.

The frontend exposes per-domain mirror lists (GitHub proxy, HuggingFace,
PyPI). The user clicks "测速" and we return the latency-sorted result so the
UI can offer a one-click "use the fastest" action.

Probes are HEAD requests with a strict timeout, run concurrently in a thread
pool. We cache nothing — the user is asking for "fresh now" — but we cap the
concurrent fan-out so a slow link doesn't snowball into a hundred parallel
sockets.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# Per-host built-in presets. The frontend can either send its own urls or
# ask for a category by name. Keep the GitHub proxy list as host roots
# (no path) so we can probe with cheap HEAD requests.
PRESETS: dict[str, list[dict[str, str]]] = {
    "github_proxy": [
        {"label": "直连 (无代理)", "value": "", "probe": "https://github.com/"},
        {"label": "gh-proxy.org", "value": "https://gh-proxy.org", "probe": "https://gh-proxy.org/"},
        {"label": "hk.gh-proxy.org", "value": "https://hk.gh-proxy.org", "probe": "https://hk.gh-proxy.org/"},
        {"label": "cdn.gh-proxy.org", "value": "https://cdn.gh-proxy.org", "probe": "https://cdn.gh-proxy.org/"},
        {"label": "edgeone.gh-proxy.org", "value": "https://edgeone.gh-proxy.org", "probe": "https://edgeone.gh-proxy.org/"},
        {"label": "ghfast.top", "value": "https://ghfast.top", "probe": "https://ghfast.top/"},
        {"label": "mirror.ghproxy.com", "value": "https://mirror.ghproxy.com", "probe": "https://mirror.ghproxy.com/"},
        {"label": "ghproxy.net", "value": "https://ghproxy.net", "probe": "https://ghproxy.net/"},
        {"label": "kkgithub.com", "value": "https://kkgithub.com", "probe": "https://kkgithub.com/"},
    ],
    "huggingface": [
        {"label": "huggingface.co (官方)", "value": "", "probe": "https://huggingface.co/"},
        {"label": "hf-mirror.com", "value": "https://hf-mirror.com", "probe": "https://hf-mirror.com/"},
        {"label": "modelscope.cn (备选)", "value": "https://modelscope.cn", "probe": "https://modelscope.cn/"},
    ],
    "pypi": [
        {"label": "pypi.org (官方)", "value": "https://pypi.org/simple/", "probe": "https://pypi.org/simple/"},
        {"label": "TUNA 清华", "value": "https://pypi.tuna.tsinghua.edu.cn/simple", "probe": "https://pypi.tuna.tsinghua.edu.cn/simple/"},
        {"label": "中科大 USTC", "value": "https://pypi.mirrors.ustc.edu.cn/simple", "probe": "https://pypi.mirrors.ustc.edu.cn/simple/"},
        {"label": "阿里云", "value": "https://mirrors.aliyun.com/pypi/simple", "probe": "https://mirrors.aliyun.com/pypi/simple/"},
        {"label": "腾讯云", "value": "https://mirrors.cloud.tencent.com/pypi/simple", "probe": "https://mirrors.cloud.tencent.com/pypi/simple/"},
        {"label": "华为云", "value": "https://mirrors.huaweicloud.com/repository/pypi/simple", "probe": "https://mirrors.huaweicloud.com/repository/pypi/simple/"},
        {"label": "豆瓣", "value": "https://pypi.douban.com/simple", "probe": "https://pypi.douban.com/simple/"},
        {"label": "网易", "value": "https://mirrors.163.com/pypi/simple", "probe": "https://mirrors.163.com/pypi/simple/"},
    ],
}


class ProbeRequest(BaseModel):
    category: str | None = None  # one of the keys in PRESETS, optional
    urls: list[str] | None = None  # extra ad-hoc URLs to probe
    timeout_ms: int = 4000


class ProbeResult(BaseModel):
    label: str
    value: str  # what to write into Settings (may be "" for direct)
    probe: str  # what we actually requested
    ok: bool
    status: int | None = None
    latency_ms: float | None = None
    error: str | None = None


def _probe_one(url: str, timeout: float) -> tuple[int | None, float | None, str | None]:
    """Send a HEAD; fall back to GET if the host doesn't allow HEAD."""
    headers = {"User-Agent": "lorahub-network-probe/0.2"}
    started = time.monotonic()
    try:
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            elapsed = (time.monotonic() - started) * 1000.0
            return resp.status, elapsed, None
    except urllib.error.HTTPError as e:
        # 4xx is still a valid latency signal — host is reachable.
        elapsed = (time.monotonic() - started) * 1000.0
        if 400 <= e.code < 500:
            return e.code, elapsed, None
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                resp.read(1)  # short read so we don't hang on giant payloads
                elapsed = (time.monotonic() - started) * 1000.0
                return resp.status, elapsed, None
        except Exception as inner:  # noqa: BLE001
            return None, None, str(inner)[:200]
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)[:200]


@router.get("/network/presets")
def list_presets() -> dict[str, list[dict[str, str]]]:
    return PRESETS


@router.post("/network/probe", response_model=list[ProbeResult])
def probe(req: ProbeRequest) -> list[ProbeResult]:
    targets: list[dict[str, str]] = []
    if req.category:
        if req.category not in PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown category {req.category!r}; "
                f"valid: {sorted(PRESETS)}",
            )
        targets.extend(PRESETS[req.category])
    if req.urls:
        for u in req.urls:
            targets.append({"label": u, "value": u, "probe": u})

    if not targets:
        raise HTTPException(status_code=400, detail="no targets to probe")

    timeout = max(0.5, req.timeout_ms / 1000.0)
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        futs = {
            pool.submit(_probe_one, t["probe"], timeout): t for t in targets
        }
        for fut in as_completed(futs):
            t = futs[fut]
            status, latency, err = fut.result()
            results.append(
                ProbeResult(
                    label=t.get("label", t["probe"]),
                    value=t.get("value", t["probe"]),
                    probe=t["probe"],
                    ok=status is not None,
                    status=status,
                    latency_ms=round(latency, 1) if latency is not None else None,
                    error=err,
                )
            )

    # Sort: reachable first, then by latency ascending; unreachable to the bottom.
    results.sort(
        key=lambda r: (
            not r.ok,
            r.latency_ms if r.latency_ms is not None else float("inf"),
        )
    )
    return results
