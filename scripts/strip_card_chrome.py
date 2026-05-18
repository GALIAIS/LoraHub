"""Strip the redundant `rounded-[6px] border-border/(60|70) shadow-[var(--panel-shadow)]`
chrome class from every <Card>. The shiro-surface CSS class on Card now
provides border + radius + shadow, so callers can drop the duplicate
utility classes."""
import re
from pathlib import Path

PATTERNS = [
    # The full chrome triple seen in dashboard / settings / files-tab.
    re.compile(r"rounded-\[6px\] border-border/(?:60|70) shadow-\[var\(--panel-shadow\)\]\s*"),
    # Shorter variant used by the analysis panels (metric grid / ai-card
    # / series-stats / metrics-table / samples-gallery / workbench main).
    # `shiro-surface` already provides border + radius so this pair is
    # redundant. We only target it inside <Card …>; freestanding divs
    # that emulate Card chrome (filter chips, modal shells) keep theirs.
    re.compile(r"rounded-\[6px\] border-border/(?:60|70)\s*"),
]
# After the chrome strip, some Card calls are left with an empty
# `className=""` attribute. Drop those too — pure noise.
EMPTY_CLASS_NAME = re.compile(r'\s*className=""\s*')

# Optional: also pick up `bg-card/80` adjacent to the chrome, which the
# preflight panel uses; but that's a real opaque-fill choice we shouldn't
# strip. Leave it.

ROOT = Path(__file__).resolve().parents[1] / "web" / "src"
files = list(ROOT.rglob("*.tsx"))
changed = 0
total_removals = 0
for f in files:
    text = f.read_text(encoding="utf-8")
    new = text
    file_removed = 0
    for pat in PATTERNS:
        new, n = pat.subn("", new)
        file_removed += n
    # Empty `className=""` left over from above; only inside <Card ...>.
    new, n_empty = re.subn(
        r"(<Card)\s+className=\"\"(\s*)", r"\1\2", new
    )
    file_removed += n_empty
    if new != text:
        f.write_text(new, encoding="utf-8")
        changed += 1
        total_removals += file_removed
        print(f"  {f.relative_to(ROOT)}: -{file_removed}")
print(f"\ntotal: changed {changed} files, removed {total_removals} occurrences")
