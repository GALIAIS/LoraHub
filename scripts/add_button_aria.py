"""F9 — auto-add `aria-label` whenever a <Button … title="…"> exists
without one. The title is meant for sighted users hovering with a
mouse; aria-label is what assistive tech reads. They almost always
want the same string, so this is a mechanical one-liner per match.

We deliberately scope to <Button> elements (the ui primitive); plain
<button> already has accessible-name fallback heuristics in some
browsers and we avoid touching them to limit blast radius.
"""
import re
from pathlib import Path

# Match an opening <Button …> that has a `title="…"` attribute but no
# `aria-label`. Capture the full attribute block + the title content so
# we can splice in the aria-label.
BUTTON_OPEN = re.compile(
    r"(<Button\b)((?:[^>]|\n)*?)(\s+title=\"([^\"]+)\")((?:[^>]|\n)*?>)",
    re.DOTALL,
)


def needs_aria(attrs: str) -> bool:
    return "aria-label" not in attrs


def add_aria(match: re.Match[str]) -> str:
    name = match.group(1)
    pre_attrs = match.group(2)
    title_attr = match.group(3)
    title_text = match.group(4)
    post_attrs = match.group(5)
    # If aria-label already present anywhere in the open tag, don't touch.
    if not needs_aria(pre_attrs + title_attr + post_attrs):
        return match.group(0)
    inserted = f' aria-label="{title_text}"'
    return f"{name}{pre_attrs}{title_attr}{inserted}{post_attrs}"


ROOT = Path(__file__).resolve().parents[1] / "web" / "src"
files = list(ROOT.rglob("*.tsx"))

changed_files = 0
total_added = 0
for f in files:
    text = f.read_text(encoding="utf-8")
    new, n = BUTTON_OPEN.subn(add_aria, text)
    # subn counts every match including ones already containing aria-
    # label (the substitution is a no-op for those). Recompute the
    # actual delta by comparing aria-label count.
    if new != text:
        before = text.count('aria-label="')
        after = new.count('aria-label="')
        delta = after - before
        if delta > 0:
            f.write_text(new, encoding="utf-8")
            changed_files += 1
            total_added += delta
            print(f"  {f.relative_to(ROOT)}: +{delta}")

print(f"\ntotal: changed {changed_files} files, added {total_added} aria-label attrs")
