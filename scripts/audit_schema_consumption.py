"""Enumerate leaf fields in TrainingConfig via pydantic model_fields,
then probe compiler usage. See B1 audit notes."""
import re
from pathlib import Path
from typing import get_args, get_origin

from pydantic import BaseModel
from lorahub.core.config.schema import TrainingConfig

repo = Path(__file__).resolve().parent
KOHYA = (
    repo / "lorahub" / "core" / "backends" / "kohya" / "compiler.py"
).read_text(encoding="utf-8")
DP = (
    repo / "lorahub" / "core" / "backends" / "diffusion_pipe" / "compiler.py"
).read_text(encoding="utf-8")


def is_model(t) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def walk(model_cls, prefix=""):
    out = []
    for name, info in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann = info.annotation
        # Unwrap Optional / Union
        origin = get_origin(ann)
        if origin is None:
            target = ann
        else:
            args = [a for a in get_args(ann) if a is not type(None)]
            target = args[0] if args else None
        # Recurse into nested BaseModel; otherwise leaf.
        if target and is_model(target):
            out.extend(walk(target, path))
        else:
            out.append(path)
    return out


def classify(field: str):
    leaf = field.split(".")[-1]
    emit_pat = re.compile(
        rf"\b(args|parts)\.append\([^)]*\b{re.escape(leaf)}\b"
    )
    track_pat = re.compile(
        rf'_track\([^)]*"{re.escape(field)}"|_track\([^)]*"[^"]*\.{re.escape(leaf)}"'
    )
    # Generic reference: any mention of `.<leaf>` or the field path.
    ref_pat = re.compile(rf"\.{re.escape(leaf)}\b")
    return {
        "kohya_emit": bool(emit_pat.search(KOHYA)),
        "kohya_warn": bool(track_pat.search(KOHYA)),
        "kohya_ref": bool(ref_pat.search(KOHYA)),
        "dp_emit": bool(emit_pat.search(DP)),
        "dp_warn": bool(track_pat.search(DP)),
        "dp_ref": bool(ref_pat.search(DP)),
    }


fields = walk(TrainingConfig)
print(f"total leaf fields: {len(fields)}")

# Categorise.
unconsumed = []  # No emit + no warn + no reference at all.
ref_only = []  # Referenced but never emit / warn (likely read for derivation).
warn_only = []  # Declared "unsupported" via _track but never emit.
emitted = []  # At least one backend emits.

for f in fields:
    c = classify(f)
    if c["kohya_emit"] or c["dp_emit"]:
        emitted.append((f, c))
    elif c["kohya_warn"] or c["dp_warn"]:
        warn_only.append((f, c))
    elif c["kohya_ref"] or c["dp_ref"]:
        ref_only.append((f, c))
    else:
        unconsumed.append(f)

print(f"  emit by ≥1 backend: {len(emitted)}")
print(f"  warn-only (_track): {len(warn_only)}")
print(f"  referenced but no emit/warn: {len(ref_only)}")
print(f"  totally unconsumed:  {len(unconsumed)}")

if unconsumed:
    print("\n## TOTALLY UNCONSUMED (high-confidence dead):")
    for f in unconsumed:
        print(f"  {f}")

if warn_only:
    print("\n## WARN-ONLY:")
    for f, c in warn_only:
        bk = []
        if c["kohya_warn"]:
            bk.append("kohya")
        if c["dp_warn"]:
            bk.append("dp")
        print(f"  {f}  [{','.join(bk)}]")

if ref_only:
    print("\n## REF-ONLY (mentioned but no compile output):")
    for f, c in ref_only[:30]:
        bk = []
        if c["kohya_ref"]:
            bk.append("kohya")
        if c["dp_ref"]:
            bk.append("dp")
        print(f"  {f}  [{','.join(bk)}]")
    if len(ref_only) > 30:
        print(f"  ... and {len(ref_only) - 30} more")
