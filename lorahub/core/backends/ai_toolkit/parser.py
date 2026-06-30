"""Parse ai-toolkit logs into LoraHub training events."""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STEP_RE = re.compile(
    r"(?:(?P<step>\d+)\s*/\s*(?P<total>\d+)|step[=: ]+(?P<step2>\d+))"
    r".*?\bloss[:= ]+(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_SAVE_RE = re.compile(r"\bSaved checkpoint to\s+(?P<path>.+?)\s*$", re.IGNORECASE)
_TQDM_RE = re.compile(
    r"^(?:(?P<label>[^:\n]+):\s*)?"
    r"(?P<percent>\d{1,3})%\|.*?\|\s*"
    r"(?P<done>[\d.]+[KMGTPE]?)/(?P<total>[\d.]+[KMGTPE]?)"
    r"(?:\s*\[(?P<elapsed>[^<,\]]+)"
    r"(?:<(?P<eta>[^,\]]+))?"
    r"(?:,\s*(?P<rate>[^\]]+))?\])?",
    re.IGNORECASE,
)
_LR_RE = re.compile(r"\blr:\s*(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)

_PHASE_LABELS = {
    "caching latents to disk": "缓存潜空间",
    "loading weights": "加载权重",
    "fetching": "下载文件",
    "downloading": "下载模型",
    "download complete": "下载完成",
}


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    stripped = _clean_line(line)
    if not stripped.strip():
        return None
    if (m := _SAVE_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.checkpoint_saved,
            payload={"path": m.group("path")},
            job_id=job_id,
        )
    if (m := _STEP_RE.search(stripped)) is not None:
        payload: dict[str, object] = {
            "step": int(m.group("step") or m.group("step2")),
            "loss": float(m.group("loss")),
        }
        if m.group("total"):
            payload["total_steps"] = int(m.group("total"))
        if lr := _extract_lr(stripped):
            payload["lr"] = lr
        if rate := _extract_tqdm_field(stripped, "rate"):
            payload["rate"] = rate
        if eta := _extract_tqdm_field(stripped, "eta"):
            payload["eta"] = eta
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)
    if (m := _TQDM_RE.search(stripped)) is not None:
        return _parse_progress(m, stripped, job_id=job_id)

    level = _level_for(stripped)
    return TrainingEvent(
        type=EventType.log,
        payload={"level": level, "message": _friendly_message(stripped)},
        job_id=job_id,
    )


def _clean_line(line: str) -> str:
    return _ANSI_RE.sub("", line).strip(" \r\n")


def _parse_progress(
    match: re.Match[str], line: str, *, job_id: str | None
) -> TrainingEvent | None:
    label = (match.group("label") or "").strip()
    lower = label.lower()
    done_raw = match.group("done")
    total_raw = match.group("total")

    if "caching latents" in lower:
        done = _parse_int(done_raw)
        total = _parse_int(total_raw)
        if done is None or total is None:
            return None
        return TrainingEvent(
            type=EventType.cache_progress,
            payload={
                "phase": "latents",
                "done": done,
                "total": total,
                "percent": int(match.group("percent")),
                "rate": (match.group("rate") or "").strip() or None,
                "eta": (match.group("eta") or "").strip() or None,
            },
            job_id=job_id,
        )

    # Bare tqdm rows after a preceding heading add noise without a phase.
    if not label:
        return None

    message = _format_progress_message(
        label=label,
        percent=int(match.group("percent")),
        done=done_raw,
        total=total_raw,
        rate=(match.group("rate") or "").strip(),
        eta=(match.group("eta") or "").strip(),
    )
    return TrainingEvent(
        type=EventType.log,
        payload={"level": "progress", "message": message, "phase": label},
        job_id=job_id,
    )


def _format_progress_message(
    *, label: str, percent: int, done: str, total: str, rate: str, eta: str
) -> str:
    lower = label.lower()
    friendly = next(
        (name for key, name in _PHASE_LABELS.items() if key in lower),
        label.strip(),
    )
    parts = [f"{friendly} {percent}% · {done}/{total}"]
    if rate:
        parts.append(rate)
    if eta:
        parts.append(f"剩余 {eta}")
    return " · ".join(parts)


def _friendly_message(line: str) -> str:
    text = line.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    if text.startswith("#"):
        return text.strip("# ").strip() or text
    if text.lower().startswith("running job:"):
        return "任务启动：" + text.split(":", 1)[1].strip()
    return text


def _extract_lr(line: str) -> float | None:
    match = _LR_RE.search(line)
    return float(match.group("lr")) if match else None


def _extract_tqdm_field(line: str, name: str) -> str | None:
    match = _TQDM_RE.search(line)
    if not match:
        return None
    value = (match.group(name) or "").strip()
    if name == "rate":
        value = value.split(", lr:", 1)[0].strip()
    return value or None


def _parse_int(value: str) -> int | None:
    if not value.isdigit():
        return None
    return int(value)


def _level_for(line: str) -> str:
    lowered = line.lower()
    if _looks_like_error(line):
        return "error"
    if "warning" in lowered or lowered.startswith("warn"):
        return "warning"
    return "info"


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    return (
        "error" in lowered
        or "traceback" in lowered
        or "out of memory" in lowered
        or "exception" in lowered
        or "runtimeerror" in lowered
    )


__all__ = ["parse_line"]
