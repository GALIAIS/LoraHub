"""Training assistant — hyperparameter suggestions + failure diagnosis.

Two independent surfaces, both pure functions over inputs the trainer
already produces (no side effects, no live model probing):

* ``recommend_hyperparams(dataset_size, gpu_vram_mb, backend, ...)``
  — suggests batch size, gradient accumulation, learning rate, rank,
  epochs, and a few flags based on a small published-results table.
  Use case: a user opens "new training run" with a dataset they just
  prepared and wants reasonable defaults without spending an hour on
  forum threads.

* ``diagnose_failure(workspace, returncode, error)`` — opens the
  workspace's ``events.jsonl`` + last few hundred lines of the
  training log and tries to classify the failure mode (OOM, NaN
  loss, missing dependency, dataset corruption, hardware issue,
  user abort, …). Every classification carries a remediation hint.
  Use case: a user comes back to a red badge in the UI and wants to
  know "what now?" in one click.

Both functions are deliberately heuristic — we'd rather give a 70%
correct recommendation that's actionable than a 100% accurate one
that takes ten minutes to derive. The caller can layer their own
overrides on top.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from lorahub.api import diagnosis_patterns as _diagnosis_patterns

logger = logging.getLogger(__name__)

BackendName = Literal["kohya", "diffusion-pipe", "anima_lora"]


# --------------------------------------------------------------------------- #
# Hyperparameter recommendations                                              #
# --------------------------------------------------------------------------- #


@dataclass
class HyperparamSuggestion:
    """One coherent suggested config — easy to merge into existing presets."""

    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    network_dim: int
    network_alpha: int
    max_train_epochs: int
    optimizer_type: str
    extra_flags: dict[str, Any] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_hyperparams(
    *,
    dataset_size: int,
    gpu_vram_mb: int,
    backend: BackendName = "anima_lora",
    target: Literal["character", "style", "concept"] = "character",
) -> HyperparamSuggestion:
    """Heuristic hyperparam recommendation.

    The numbers come from observed-good defaults across the kohya / dp /
    anima communities — they're starting points, not endpoints. The
    rationale list explains *why* each value was picked so the user can
    judge whether their case differs.
    """
    rationale: list[str] = []

    # ---- batch size + gradient accumulation ----------------------------
    # Anchor on VRAM. Anima at bf16 + gradient_checkpointing fits one
    # 1024² sample per ~6GB of free VRAM. Below that we have to lean on
    # gradient_accumulation to reach an effective batch >= 4 (the
    # threshold below which BatchNorm-free networks start to overfit
    # statistics on tiny datasets).
    if gpu_vram_mb >= 24_000:
        batch_size, ga = 4, 1
        rationale.append(
            f"VRAM {gpu_vram_mb / 1024:.0f}GB ≥ 24GB → batch=4, no grad accum"
        )
    elif gpu_vram_mb >= 16_000:
        batch_size, ga = 2, 2
        rationale.append(
            f"VRAM {gpu_vram_mb / 1024:.0f}GB → batch=2, ga=2 (effective batch 4)"
        )
    elif gpu_vram_mb >= 12_000:
        batch_size, ga = 1, 4
        rationale.append(
            f"VRAM {gpu_vram_mb / 1024:.0f}GB → batch=1, ga=4 (use --gradient_checkpointing)"
        )
    else:
        batch_size, ga = 1, 8
        rationale.append(
            f"VRAM {gpu_vram_mb / 1024:.0f}GB tight → batch=1, ga=8, "
            "consider blocks_to_swap on anima"
        )

    # ---- learning rate -------------------------------------------------
    # Per-target LRs from "what doesn't blow up" community surveys.
    # Effective batch (batch_size * ga) interacts with LR linearly —
    # we keep LR ~constant across batch sizes since gradient accumulation
    # already averages.
    base_lr_by_target = {
        "character": 1e-4,
        "style": 5e-5,
        "concept": 8e-5,
    }
    lr = base_lr_by_target[target]
    rationale.append(
        f"target={target} → unet_lr {lr:.0e} (community-surveyed sweet spot)"
    )

    # ---- network rank --------------------------------------------------
    # rank scales with how much "new behaviour" the LoRA needs to learn.
    # For 50-300 image character LoRAs r=16 is plenty; small datasets
    # benefit from r=8 to cap capacity (less overfitting); large/style
    # datasets want r=32+.
    if dataset_size < 50:
        network_dim = 8
        rationale.append(f"dataset_size={dataset_size} < 50 → network_dim=8 (cap capacity)")
    elif dataset_size < 300:
        network_dim = 16
        rationale.append(
            f"dataset_size={dataset_size} → network_dim=16 (typical character size)"
        )
    elif dataset_size < 1000:
        network_dim = 32
        rationale.append(f"dataset_size={dataset_size} → network_dim=32 (style range)")
    else:
        network_dim = 64
        rationale.append(
            f"dataset_size={dataset_size} large → network_dim=64 (more headroom)"
        )

    # alpha = network_dim is a stable default. alpha < dim regularises
    # at the cost of effective LR (need to bump LR ~×(dim/alpha)).
    network_alpha = network_dim

    # ---- epochs --------------------------------------------------------
    # Total samples seen ≈ epochs × dataset_size. Anima recipes target
    # 1k-10k effective samples for character; we land in that range.
    target_samples = {
        "character": 5000,
        "style": 8000,
        "concept": 6000,
    }[target]
    max_train_epochs = max(8, min(64, round(target_samples / max(dataset_size, 1))))
    rationale.append(
        f"target_samples={target_samples} / size={dataset_size} → epochs≈{max_train_epochs}"
    )

    # ---- optimizer -----------------------------------------------------
    if gpu_vram_mb < 12_000:
        optimizer_type = "AdamW8bit"
        rationale.append("low VRAM → AdamW8bit (saves ~4GB optimizer state)")
    else:
        optimizer_type = "AdamW"
        rationale.append("ample VRAM → AdamW (highest quality default)")

    # ---- extras --------------------------------------------------------
    extra: dict[str, Any] = {}
    if gpu_vram_mb < 16_000:
        extra["gradient_checkpointing"] = True
        rationale.append("VRAM-bound → gradient_checkpointing on")
    if backend == "anima_lora":
        # Anima's stable defaults: logit-normal sampling + min-SNR-γ for
        # convergence speed; EMA + nan_guard for safety.
        extra["weighting_scheme"] = "min_snr_rf"
        extra["min_snr_gamma"] = 5
        extra["ema"] = True
        extra["nan_guard"] = True
        rationale.append(
            "anima: enabling min_snr_rf + ema + nan_guard (no extra cost, "
            "safer + faster convergence)"
        )

    return HyperparamSuggestion(
        batch_size=batch_size,
        gradient_accumulation_steps=ga,
        learning_rate=lr,
        network_dim=network_dim,
        network_alpha=network_alpha,
        max_train_epochs=max_train_epochs,
        optimizer_type=optimizer_type,
        extra_flags=extra,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Failure diagnosis                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class DiagnosisFinding:
    """One classifier hit. Multiple findings may apply to the same run."""

    category: str  # short tag like "oom" / "nan" / "missing_dep"
    severity: Literal["info", "warn", "error"]
    message: str
    remediation: str
    evidence: str = ""  # short excerpt from the log/event that triggered

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Patterns ordered by priority — first match wins for each category.
# Anchored on stable strings observed in actual failed runs (kohya /
# diffusers / accelerate / anima). Add new patterns here when a new
# failure mode shows up; the data lives next to the matcher so a UI
# tooltip can quote the exact regex.
_PATTERNS: list[tuple[str, str, Literal["info", "warn", "error"], str, str]] = (
    # The catalogue lives in lorahub.api.diagnosis_patterns so the
    # streaming WARN watcher (Phase 3) can share the same regexes.
    # Keep `_PATTERNS` as a module-level alias so existing tests /
    # call sites that monkeypatch it still work.
    list(_diagnosis_patterns.get_patterns())
)


def diagnose_failure(
    workspace: Path | str,
    returncode: int | None = None,
    error: str | None = None,
    *,
    log_lines: int = 400,
) -> dict[str, Any]:
    """Open the workspace's training log + events.jsonl and classify.

    Output:
      ``{"findings": [...], "summary": "...", "log_excerpt": "..."}``

    where ``findings`` is a list of :class:`DiagnosisFinding` dicts.
    The matcher is best-effort — a job that produces zero findings
    isn't necessarily healthy, just unfamiliar; ``summary`` always
    carries something pointing the user at the workspace path.
    """
    workspace = Path(workspace)
    findings: list[DiagnosisFinding] = []
    log_excerpt = ""

    # --- Pull whatever signal we have.
    error_text = error or ""
    log_text = ""
    log_path = _find_training_log(workspace)
    if log_path is not None:
        try:
            log_text = _tail_text(log_path, log_lines)
            log_excerpt = log_text[-2000:]
        except OSError as exc:
            logger.debug("could not read %s: %s", log_path, exc)

    events = _read_events(workspace, max_count=200)

    haystack = "\n".join(
        s for s in (error_text, log_text, _events_summary(events)) if s
    )

    # --- Pattern match.
    seen_categories: set[str] = set()
    for category, pattern, severity, message, remediation in _PATTERNS:
        if category in seen_categories:
            continue
        m = re.search(pattern, haystack, flags=re.IGNORECASE)
        if not m:
            continue
        seen_categories.add(category)
        evidence_line = _line_around(haystack, m.start())
        findings.append(
            DiagnosisFinding(
                category=category,
                severity=severity,
                message=message,
                remediation=remediation,
                evidence=evidence_line,
            )
        )

    # --- Fallback summaries when nothing matched.
    if not findings:
        if returncode is None:
            summary = "Job finished cleanly (no diagnostic signal needed)."
        elif returncode == 0:
            summary = "Job exited with status 0; no failure to diagnose."
        else:
            summary = (
                f"Job exited with non-zero status {returncode} but no known "
                "failure pattern matched. Check the log tail for the "
                "actual traceback."
            )
            findings.append(
                DiagnosisFinding(
                    category="unknown",
                    severity="warn",
                    message=summary,
                    remediation="Open the workspace and inspect events.jsonl + "
                    "training.log manually.",
                    evidence=log_excerpt[-400:],
                )
            )
    else:
        # Pick the highest-severity finding for the headline.
        severity_rank = {"error": 3, "warn": 2, "info": 1}
        head = max(findings, key=lambda f: severity_rank.get(f.severity, 0))
        summary = head.message

    return {
        "findings": [f.to_dict() for f in findings],
        "summary": summary,
        "log_excerpt": log_excerpt,
        "log_path": str(log_path) if log_path is not None else None,
    }


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _find_training_log(workspace: Path) -> Path | None:
    """Look for a training log under common paths.

    Different backends drop their stdout/stderr at different filenames
    (kohya: training.log; anima: train.log or output/<run>/train.log).
    We try a small candidate list rather than indexing the entire dir
    so a giant workspace doesn't get walked on every diagnose call.
    """
    candidates = [
        workspace / "training.log",
        workspace / "train.log",
        workspace / "stdout.log",
        workspace / "console.log",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _tail_text(path: Path, lines: int) -> str:
    """Cheap reverse-tail for text files. Avoids loading 10MB log
    files just to grep their last 400 lines."""
    block = 64 * 1024
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        data = b""
        pos = size
        while pos > 0 and data.count(b"\n") <= lines:
            read = min(block, pos)
            pos -= read
            fh.seek(pos)
            data = fh.read(read) + data
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _read_events(workspace: Path, *, max_count: int) -> list[dict[str, Any]]:
    """Pull the trailing events from ``events.jsonl`` if present."""
    p = workspace / "events.jsonl"
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out[-max_count:]


def _events_summary(events: list[dict[str, Any]]) -> str:
    """Project events down to the strings the regex matcher cares about."""
    parts: list[str] = []
    for ev in events:
        for key in ("event_type", "message", "error", "name", "type"):
            v = ev.get(key)
            if isinstance(v, str):
                parts.append(v)
    return "\n".join(parts)


def _line_around(text: str, idx: int, *, window: int = 160) -> str:
    """Single line around offset ``idx`` for evidence quoting."""
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    snippet = text[start:end]
    # Trim partial lines on the edges so the quote doesn't end mid-word.
    if start > 0 and "\n" in snippet:
        snippet = snippet[snippet.index("\n") + 1 :]
    if end < len(text) and "\n" in snippet:
        snippet = snippet[: snippet.rindex("\n")]
    return snippet.strip()


__all__ = [
    "BackendName",
    "DiagnosisFinding",
    "HyperparamSuggestion",
    "diagnose_failure",
    "recommend_hyperparams",
]
