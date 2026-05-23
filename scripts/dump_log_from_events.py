"""Reconstruct a flat training log from a workspace's ``events.jsonl``.

LoraHub's ``SubprocessRunner`` stores trainer stdout/stderr as
``type='log'`` entries in ``events.jsonl`` rather than writing a
plaintext log file. When you need the *raw* trainer output (Python
traceback, accelerate launcher banner, sd-scripts progress bars), this
script walks the jsonl in event order and prints each ``log`` event's
``message`` line, with the ``source`` (``stdout`` / ``stderr`` /
``runner`` / ``preprocess``) shown as a column prefix.

Usage:
    python scripts/dump_log_from_events.py <workspace>
    python scripts/dump_log_from_events.py <workspace> --stderr-only
    python scripts/dump_log_from_events.py <workspace> -o training.log

Exit codes:
    0  events.jsonl read, output written.
    2  workspace path or events.jsonl missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator


def _iter_log_events(events_path: Path) -> Iterator[dict[str, object]]:
    """Yield every ``type='log'`` row, skipping malformed lines.

    A malformed jsonl line (truncated write during a kill -9, mixed
    encoding) shouldn't sink the whole dump — we'd rather show the
    user N-1 good lines than nothing.
    """
    with events_path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("type") == "log":
                yield row


def _format_row(row: dict[str, object], *, show_time: bool) -> str:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    assert isinstance(payload, dict)
    message = str(payload.get("message", ""))
    source = str(payload.get("source", "?"))
    level = str(payload.get("level", "info"))

    parts: list[str] = []
    if show_time:
        ts = row.get("timestamp")
        if isinstance(ts, (int, float)):
            stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            parts.append(stamp)
    # Pad source so columns align — sd-scripts traces are easier to read
    # when stderr lines stay in their own column.
    parts.append(f"[{source:<10}]")
    if level not in {"info", ""}:
        parts.append(f"({level})")
    parts.append(message)
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "workspace",
        type=Path,
        help="Run workspace, e.g. runs/your_style_anima_8gb-20260523-022822",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the reconstructed log here (default: stdout).",
    )
    parser.add_argument(
        "--stderr-only",
        action="store_true",
        help="Print only events where source=stderr — usually the failure trace.",
    )
    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Skip the wall-clock prefix.",
    )
    args = parser.parse_args(argv)

    workspace: Path = args.workspace
    if not workspace.is_dir():
        print(f"workspace not a directory: {workspace}", file=sys.stderr)
        return 2
    events_path = workspace / "events.jsonl"
    if not events_path.is_file():
        print(f"no events.jsonl under {workspace}", file=sys.stderr)
        return 2

    out_fh = (
        args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    )
    try:
        written = 0
        for row in _iter_log_events(events_path):
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            assert isinstance(payload, dict)
            if args.stderr_only and payload.get("source") != "stderr":
                continue
            line = _format_row(row, show_time=not args.no_time)
            out_fh.write(line + "\n")
            written += 1
        if args.output:
            print(f"wrote {written} log line(s) -> {args.output}", file=sys.stderr)
    finally:
        if args.output:
            out_fh.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
