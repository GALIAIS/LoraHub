"""Streaming pattern watcher for trainer subprocess output.

The post-mortem path (``training_assistant.diagnose_failure``) only
runs once the job exits, so the user sees no signal during a run that
prints a clear error and then stalls for 30 minutes before crashing.

This watcher is the live counterpart: it sees every stdout/stderr line
as the runner pumps it, applies the same regex catalogue, and emits a
``diagnostic_warning`` event the first time each rule fires. A 60-second
suppression window per category prevents log spam if (e.g.) torch keeps
re-emitting the same OOM line during teardown.

Design rules:
- Pure, no I/O. The runner owns the subprocess and the event sink.
- Stateful per-instance: the dedup cache is bound to one watcher, so a
  fresh job starts with a clean slate.
- Compiled regexes cached at construction. A long run can pump tens of
  thousands of lines and recompiling on every line would burn measurable
  CPU.
- One match per line, max — we pick the *highest-severity* category so
  a generic ``subprocess_returncode`` warn never out-shouts an
  ``oom`` error on the same line.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from lorahub.api.diagnosis_patterns import Severity, get_patterns
from lorahub.core.events import EventType, TrainingEvent


_SEVERITY_RANK: dict[Severity, int] = {"error": 3, "warn": 2, "info": 1}

# Don't re-fire the same category more than once per this many seconds.
# 60s is wide enough to suppress noisy teardown loops without hiding a
# legitimate second incident an hour later.
_SUPPRESS_WINDOW_S = 60.0


@dataclass(slots=True, frozen=True)
class _CompiledRule:
    category: str
    pattern: re.Pattern[str]
    severity: Severity
    message: str
    remediation: str


def _compile_rules() -> list[_CompiledRule]:
    out: list[_CompiledRule] = []
    for category, pattern, severity, message, remediation in get_patterns():
        out.append(
            _CompiledRule(
                category=category,
                pattern=re.compile(pattern, flags=re.IGNORECASE),
                severity=severity,
                message=message,
                remediation=remediation,
            )
        )
    return out


# Module-level cache so repeated runner instantiations don't recompile
# the same ~22 regexes. Tests can blow it away by re-importing.
_DEFAULT_RULES: tuple[_CompiledRule, ...] | None = None


def _default_rules() -> tuple[_CompiledRule, ...]:
    global _DEFAULT_RULES
    if _DEFAULT_RULES is None:
        _DEFAULT_RULES = tuple(_compile_rules())
    return _DEFAULT_RULES


class StreamingDiagnosticWatcher:
    """Examine trainer log lines in real time and emit findings as events."""

    def __init__(
        self,
        on_event: Callable[[TrainingEvent], None],
        *,
        job_id: str | None = None,
        rules: tuple[_CompiledRule, ...] | None = None,
        clock: Callable[[], float] = time.time,
        suppress_window_s: float = _SUPPRESS_WINDOW_S,
    ) -> None:
        self._on_event = on_event
        self._job_id = job_id
        self._rules = rules if rules is not None else _default_rules()
        self._clock = clock
        self._suppress_window_s = suppress_window_s
        # category -> last-emit timestamp
        self._last_emit: dict[str, float] = {}

    def feed(self, line: str, source: str = "stdout") -> None:
        """Examine one log line. Emits at most one diagnostic_warning event.

        ``source`` is propagated into the event payload so consumers can
        tell stdout from stderr matches; both are checked because some
        backends route trainer warnings through stdout.
        """
        if not line:
            return
        # The runner already strips trailing newline + ``replace`` errors,
        # but logs that come from sd-scripts on Windows can include
        # trailing \r — strip defensively.
        clean = line.rstrip("\r\n")
        if not clean:
            return

        best: _CompiledRule | None = None
        best_match: re.Match[str] | None = None
        best_rank = -1
        for rule in self._rules:
            m = rule.pattern.search(clean)
            if m is None:
                continue
            rank = _SEVERITY_RANK.get(rule.severity, 0)
            if rank > best_rank:
                best = rule
                best_match = m
                best_rank = rank

        if best is None or best_match is None:
            return

        now = self._clock()
        last = self._last_emit.get(best.category)
        if last is not None and now - last < self._suppress_window_s:
            return
        self._last_emit[best.category] = now

        self._on_event(
            TrainingEvent(
                type=EventType.diagnostic_warning,
                payload={
                    "category": best.category,
                    "severity": best.severity,
                    "message": best.message,
                    "remediation": best.remediation,
                    "evidence": clean[:500],
                    "source": source,
                },
                job_id=self._job_id,
            )
        )


__all__ = ["StreamingDiagnosticWatcher"]
