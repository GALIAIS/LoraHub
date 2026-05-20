"""Normalize line endings of every .bat in scripts/ to CRLF.

cmd.exe on Windows reads batch files line-by-line based on CR + LF; LF-only
files end up with the entire content read as one mega-line, which then gets
split on whitespace, producing garbage tokens like 'tlocal' from
'setlocal enabledelayedexpansion'. Used as a build-time cleanup so a stray
LF-saving editor never reintroduces the bug.
"""

from __future__ import annotations

import sys
from pathlib import Path


def normalize(p: Path) -> bool:
    raw = p.read_bytes()
    fixed = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if fixed == raw:
        return False
    p.write_bytes(fixed)
    return True


def main() -> int:
    scripts = Path(__file__).resolve().parent
    changed = []
    for bat in scripts.glob("*.bat"):
        if normalize(bat):
            changed.append(bat.name)
    if changed:
        print("normalized:", ", ".join(changed))
    else:
        print("nothing to normalize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
