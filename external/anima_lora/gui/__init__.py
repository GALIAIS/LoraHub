"""Anima LoRA — PySide6 GUI package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import toml
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QWidget,
)

from gui.i18n import t

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
# User-created variants live alongside the curated gui-methods files but in
# their own subdirectory so they're easy to find and don't pollute the
# built-in family list.
CUSTOM_VARIANTS_DIR = GUI_METHODS_DIR / "custom"


_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "reft",
    "postfix",
    "fera",
    "chimera",
    "ip_adapter",
    "easycontrol",
)

# Built-in variant families are discovered from each gui-methods/*.toml file's
# ``[variant]`` table (``family`` / ``label`` / ``description`` / optional
# ``order`` / ``experimental``). The hand-curated _FAMILY_VARIANTS map was
# retired in the Track 2 refactor — adding or renaming a variant is now a
# one-file change.
#
# Display order within a family is ``[variant].order`` (ascending; ties broken
# by file stem). Family ordering in the method combo stays curated via
# ``_METHOD_ORDER`` so we can keep training-only families (e.g. ``soft_tokens``)
# off the GUI without renaming files. Customs under
# ``configs/gui-methods/custom/`` are intentionally permissive — they don't
# need a ``[variant]`` block and are surfaced under every family the same way
# they were before.


def _read_variant_metadata(path: Path) -> dict:
    """Return the ``[variant]`` table from a gui-methods TOML, or ``{}``.

    Failures (missing file, parse error, missing table) yield an empty dict
    so callers can treat "no metadata" uniformly — built-in validation is
    handled by ``tests/test_gui_variants.py``, not here.
    """
    if not path.is_file():
        return {}
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str, str]]]:
    """Map family → list of (order, stem, label) tuples for built-in variants.

    Built-in = directly under ``configs/gui-methods/`` (not the ``custom/``
    subdir). Files without a ``[variant].family`` are dropped silently —
    they're either malformed or intentionally hidden, and listing them under
    a guessed family would just re-introduce the stale-map problem.
    """
    by_family: dict[str, list[tuple[int, str, str]]] = {}
    if not GUI_METHODS_DIR.is_dir():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        label = meta.get("label") if isinstance(meta.get("label"), str) else path.stem
        by_family.setdefault(family, []).append((order_int, path.stem, label))
    for entries in by_family.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return by_family


def variant_metadata(variant: str) -> dict:
    """Return the ``[variant]`` metadata for a built-in or ``custom/<name>``
    variant. Empty dict when the file has no ``[variant]`` block (custom
    variants may legitimately omit it)."""
    return _read_variant_metadata(variant_path(variant))


def list_methods() -> list[str]:
    """Method families, in a user-friendly order (lora first)."""
    return list(_METHOD_ORDER)


def list_gui_variants(method: str) -> list[str]:
    """gui-methods/*.toml files for the method family + all user customs.

    Built-in variants are filtered to those whose ``[variant].family`` matches
    ``method``, sorted by ``[variant].order`` then by file stem. Custom
    variants in ``configs/gui-methods/custom/*.toml`` are surfaced for every
    family — users name them freely and we don't try to bind a file to a
    specific family.
    """
    by_family = _builtin_variants_by_family()
    ordered = [stem for _order, stem, _label in by_family.get(method, [])]
    if CUSTOM_VARIANTS_DIR.exists():
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            ordered.append(f"custom/{p.stem}")
    return ordered


def is_custom_variant(name: str) -> bool:
    return name.startswith("custom/")


def custom_variant_path(name: str) -> Path:
    """Resolve 'custom/<name>' (or bare '<name>') to the on-disk file path."""
    stem = name[len("custom/") :] if name.startswith("custom/") else name
    return CUSTOM_VARIANTS_DIR / f"{stem}.toml"


def variant_path(variant: str) -> Path:
    """Resolve a variant identifier (built-in or 'custom/<name>') to its file."""
    return GUI_METHODS_DIR / f"{variant}.toml"


def _load_all_presets() -> dict:
    """Built-in sections in ``configs/presets.toml`` plus user-created flat
    files under ``configs/custom/<name>.toml`` (one preset per file)."""
    presets: dict = {}
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    if CUSTOM_DIR.exists():
        for p in sorted(CUSTOM_DIR.glob("*.toml")):
            try:
                presets[p.stem] = toml.loads(p.read_text(encoding="utf-8"))
            except (toml.TomlDecodeError, OSError):
                continue
    return presets


def list_presets() -> list[str]:
    return sorted(_load_all_presets())


def is_custom_preset(name: str) -> bool:
    return (CUSTOM_DIR / f"{name}.toml").exists()


def custom_preset_path(name: str) -> Path:
    return CUSTOM_DIR / f"{name}.toml"


_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "use_ortho",
        "use_timestep_mask",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        "add_reft",
        "min_rank",
        "alpha_rank_scale",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "reft_dim",
        "reft_alpha",
        "reft_layers",
        "sigma_feature_dim",
        "router_targets",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        "network_train_unet_only",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "gradient_accumulation_steps",
        "use_shuffled_caption_variants",
        "caption_dropout_rate",
        "optimizer_type",
        "lr_scheduler",
        "timestep_sampling",
        "discrete_flow_shift",
        "use_valid",
        "validation_split_num",
    },
    "Performance": {
        "attn_mode",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "blocks_to_swap",
        "torch_compile",
        "compile_mode",
        "trim_crossattn_kv",
        "cache_llm_adapter_outputs",
        "masked_loss",
        "mixed_precision",
        "static_token_count",
        "vae_chunk_size",
        "vae_disable_cache",
        "cache_latents",
        "cache_latents_to_disk",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "skip_cache_check",
        "layer_start",
        "use_cmmd",
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "output_dir",
        "output_name",
        "save_model_as",
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "path_pattern",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
_SKIP = {"base_config", "dataset_config", "general", "datasets", "variant"}

# Virtual keys appear in the form like normal fields but don't round-trip as
# flat TOML keys — they're derived from / written into structured sections
# (e.g. ``use_valid`` toggles a `[[datasets]]` validation_split_num override).
# The save loop in ConfigTab skips these, and per-key apply helpers handle the
# structured write.
_VIRTUAL_KEYS = {"use_valid", "validation_split_num"}

# Fields shown under the "Basic" section. Everything else falls under the
# collapsible "Advanced" section. Picked to cover the knobs a first-time user
# realistically wants to touch (rate/length/output, headline architecture
# size, headline VRAM knobs, dataset/output paths) without exposing the long
# tail of regularizer / router / adapter-internal parameters.
_BASIC = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "output_name",
    "use_shuffled_caption_variants",
    "caption_dropout_rate",
    "gradient_checkpointing",
    "blocks_to_swap",
    "source_image_dir",
    "lora_cache_dir",
    "output_dir",
    "path_pattern",
    "use_valid",
    "validation_split_num",
}


def is_basic_field(key: str) -> bool:
    return key in _BASIC


# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


# ── Helpers ────────────────────────────────────────────────────


def _load(p: Path) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(p: Path, d: dict):
    p.write_text(toml.dumps(d), encoding="utf-8")


def merged_method_preset(method: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Return (merged_dict, origin_map). origin_map[key] is 'base' | 'preset' | 'method'."""
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    meth = _load(METHODS_DIR / f"{method}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"
    return merged, origin


def merged_gui_variant_preset(variant: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml. The GUI uses this
    instead of `merged_method_preset` so edits/training target the clean
    per-variant file, not the toggle-block methods/ tree."""
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    meth = _load(GUI_METHODS_DIR / f"{variant}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"

    # Inject the `use_valid` virtual key derived from the [[datasets]] block.
    # The variant file may shallow-override base.toml's validation_split_num /
    # validation_split via _apply_dataset_overrides in library/config/io.py; we
    # surface that as a single checkbox the user can flip in the form.
    variant_override = _variant_validation_override(meth)
    if variant_override is not None:
        merged["use_valid"] = variant_override
        origin["use_valid"] = "method"
    else:
        merged["use_valid"] = _base_validation_enabled(base)
        origin["use_valid"] = "base"

    # Inject `validation_split_num` (integer) from the same [[datasets]] block.
    # Shown as a basic field so users can resize the held-out slice directly
    # without dropping to base.toml. When the variant doesn't override it, the
    # value comes from base.toml.
    variant_vsn = _variant_validation_split_num(meth)
    if variant_vsn is not None:
        merged["validation_split_num"] = variant_vsn
        origin["validation_split_num"] = "method"
    else:
        merged["validation_split_num"] = _base_validation_split_num(base)
        origin["validation_split_num"] = "base"
    return merged, origin


def _validation_enabled_from_datasets(datasets: Any) -> Optional[bool]:
    """Inspect a TOML ``[[datasets]]`` list and decide whether validation is
    enabled. Returns ``True`` / ``False`` when either validation key is
    explicitly set on the first dataset entry, or ``None`` when no override
    is present (caller falls back to the parent layer in the merge chain)."""
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    vsn = first.get("validation_split_num")
    vs = first.get("validation_split")
    if vsn is None and vs is None:
        return None
    return (vsn or 0) > 0 or (vs or 0.0) > 0.0


def _variant_validation_override(variant_data: dict) -> Optional[bool]:
    """Return the variant TOML's explicit use_valid override, or None when
    the variant doesn't touch validation_split[_num]."""
    return _validation_enabled_from_datasets(variant_data.get("datasets"))


def _base_validation_enabled(base_data: dict) -> bool:
    """Default use_valid pulled from configs/base.toml's [[datasets]] block.
    Falls back to False when the block is missing — matches the
    `validation_split == 0 and validation_split_num <= 0` short-circuit in
    library/config/loader.py:generate_dataset_group_by_blueprint."""
    return bool(_validation_enabled_from_datasets(base_data.get("datasets")))


def _validation_split_num_from_datasets(datasets: Any) -> Optional[int]:
    """Pull ``validation_split_num`` off the first [[datasets]] entry as an
    int. Returns None when the block is missing or the key isn't set."""
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    vsn = first.get("validation_split_num")
    if vsn is None:
        return None
    try:
        return int(vsn)
    except (TypeError, ValueError):
        return None


def _variant_validation_split_num(variant_data: dict) -> Optional[int]:
    """Return the variant TOML's explicit validation_split_num override, or
    None when the variant doesn't touch it."""
    return _validation_split_num_from_datasets(variant_data.get("datasets"))


def _base_validation_split_num(base_data: dict) -> int:
    """Default validation_split_num pulled from configs/base.toml. Falls back
    to 0 when the block / key is missing."""
    return _validation_split_num_from_datasets(base_data.get("datasets")) or 0


def apply_validation_choice(
    out: dict,
    enabled: bool,
    split_num: Optional[int] = None,
    base_split_num: Optional[int] = None,
) -> None:
    """Encode the use_valid checkbox (+ optional validation_split_num int)
    into the variant TOML dict ``out``.

    Enabled  → if ``split_num`` is provided and differs from ``base_split_num``,
               write {validation_split_num = split_num} on the first
               [[datasets]] entry (strips any fractional validation_split).
               Otherwise strip both keys so the base.toml value wins through
               the merge chain.
    Disabled → write {validation_split_num = 0, validation_split = 0.0} on the
               first [[datasets]] entry, creating the block if absent. This is
               applied by _apply_dataset_overrides in library/config/io.py and
               causes generate_dataset_group_by_blueprint to skip the val set.
               (The ``split_num`` int is ignored when disabled.)

    Other keys in the variant's [[datasets]] block (e.g. a custom batch_size)
    are preserved; we only touch the two validation keys."""
    existing = out.get("datasets")
    if enabled:
        keep_override = (
            split_num is not None
            and split_num > 0
            and split_num != (base_split_num or 0)
        )
        if keep_override:
            if not isinstance(existing, list):
                existing = []
                out["datasets"] = existing
            if not existing:
                existing.append({})
            first = existing[0]
            if not isinstance(first, dict):
                first = {}
                existing[0] = first
            first["validation_split_num"] = int(split_num)
            first.pop("validation_split", None)
            return
        # No override needed — strip any zero/value override so base wins.
        if not isinstance(existing, list) or not existing:
            return
        first = existing[0]
        if not isinstance(first, dict):
            return
        first.pop("validation_split_num", None)
        first.pop("validation_split", None)
        if not first and len(existing) == 1:
            del out["datasets"]
        return

    if not isinstance(existing, list):
        existing = []
        out["datasets"] = existing
    if not existing:
        existing.append({})
    first = existing[0]
    if not isinstance(first, dict):
        first = {}
        existing[0] = first
    first["validation_split_num"] = 0
    first["validation_split"] = 0.0


def confirm_resumable_checkpoint(parent: QWidget | None, merged: dict) -> bool:
    """Prompt the user when a checkpoint is on disk; return whether to launch.

    Returns True if training should proceed (Yes = let train.py auto-resume,
    No = wipe the state dir + adapter sidecar so train.py starts fresh),
    False if the user cancelled. Returns True with no prompt when there is
    nothing to resume from — the call site can wrap every train launch in
    this helper unconditionally.
    """
    found = find_resumable_checkpoint(merged)
    if found is None:
        return True
    state_dir, step = found
    choice = QMessageBox.question(
        parent,
        t("resume_checkpoint_title"),
        t("resume_checkpoint_question", step=step),
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        QMessageBox.Yes,
    )
    if choice == QMessageBox.Cancel:
        return False
    if choice == QMessageBox.Yes:
        return True
    # No → start fresh. Wipe both the state dir and the sibling
    # ``-checkpoint.safetensors`` adapter so train.py's auto_resume sees
    # nothing on disk. Bail with a warning if the deletion fails — better
    # than silently launching a resume the user explicitly opted out of.
    import shutil

    sidecar = state_dir.parent / f"{state_dir.name.removesuffix('-state')}.safetensors"
    try:
        shutil.rmtree(state_dir)
        if sidecar.is_file():
            sidecar.unlink()
    except OSError as e:
        QMessageBox.warning(
            parent,
            t("error"),
            t("resume_checkpoint_delete_failed", error=str(e)),
        )
        return False
    return True


# Cache-file suffixes written by the preprocess scripts. Kept in sync with
# preprocess/cache_latents.py, cache_text_embeddings.py, cache_pe_encoder.py.
_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"


def count_preprocess_caches(cache_dir: Path) -> dict[str, int]:
    """Count existing latent / TE / PE cache sidecars under a cache directory.

    Returns zeros (without raising) if the directory does not exist. Used to
    surface a reassurance popup that ``make preprocess`` reuses existing caches
    rather than wiping them — a recurring point of confusion for new users.

    Walks recursively so nested caches (mirroring a subfoldered source tree)
    are counted.
    """
    out = {"latents": 0, "te": 0, "pe": 0}
    if not cache_dir.is_dir():
        return out
    for p in cache_dir.rglob("*"):
        if not p.is_file():
            continue
        n = p.name
        if n.endswith(_TE_SUFFIX):
            out["te"] += 1
        elif n.endswith(_PE_SUFFIX):
            out["pe"] += 1
        elif n.endswith(_LATENT_SUFFIX):
            out["latents"] += 1
    return out


def confirm_existing_caches(
    parent: QWidget | None, cache_dir: Path, require_pe: bool = False
) -> bool:
    """Reassure the user that existing preprocess caches will be reused, not
    deleted. Returns True to proceed, False if the user cancelled.

    No-op (returns True without prompting) when the cache directory is empty
    or missing, so the call site can wrap every preprocess launch in this.
    """
    counts = count_preprocess_caches(cache_dir)
    has_any = (
        counts["latents"] > 0 or counts["te"] > 0 or (require_pe and counts["pe"] > 0)
    )
    if not has_any:
        return True

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "preprocess_existing_caches_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(t("preprocess_existing_caches_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Ok)
    return box.exec() == QMessageBox.Ok


def confirm_train_using_cache(
    parent: QWidget | None, cache_dir: Path, require_pe: bool = False
) -> bool | None:
    """Train-side cache confirmation: returns True to launch training against
    the existing cache, False if the user cancelled, or None when no cache was
    found on disk (caller should auto-chain a preprocess run instead).

    Distinct from ``confirm_existing_caches`` (which reassures during
    Preprocess) — this gates Train and exposes the empty-cache case as a
    separate ``None`` so the caller can branch into the auto-preprocess flow.
    """
    counts = count_preprocess_caches(cache_dir)
    has_any = (
        counts["latents"] > 0 or counts["te"] > 0 or (require_pe and counts["pe"] > 0)
    )
    if not has_any:
        return None

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "train_using_cache_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(t("train_using_cache_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Yes)
    return box.exec() == QMessageBox.Yes


def find_resumable_checkpoint(merged: dict) -> tuple[Path, int] | None:
    """If the merged config has a writable ``checkpointing_epochs`` and an
    on-disk checkpoint state directory exists with a usable ``train_state.json``,
    return ``(state_dir, current_step)``. Returns ``None`` when there is
    nothing to resume — that's the common case and callers should treat it as
    "just launch training normally".

    Mirrors ``library.training.checkpoints.AnimaCheckpointer.auto_resume``: the
    same ``<output_dir>/<output_name>-checkpoint-state/`` path that ``train.py``
    would auto-pick up. We deliberately do NOT enforce ``current_step <
    max_train_steps`` here — that check varies with dataset size and is
    re-evaluated at launch; the GUI prompt only needs to know "is there
    something on disk that train.py would consider resumable".
    """
    if not merged.get("checkpointing_epochs"):
        return None
    output_dir = merged.get("output_dir")
    output_name = merged.get("output_name") or "last"
    if not output_dir:
        return None
    state_dir = ROOT / output_dir / f"{output_name}-checkpoint-state"
    train_state_file = state_dir / "train_state.json"
    if not train_state_file.is_file():
        return None
    try:
        data = json.loads(train_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    step = int(data.get("current_step", 0))
    return state_dir, step


def _imgs(d: Path) -> list[Path]:
    """Return every image file under ``d`` (recursively).

    Walks subfolders so users who organize ``image_dataset/`` by character /
    series see the full pool in the browser. Cache filenames are stem-keyed
    and flat, so stems must stay unique across the tree — the trainer enforces
    this via ``_assert_unique_stems``; here we just sort and return.
    """
    if not d.exists():
        return []
    return sorted(
        p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _safetensors_in(d: Path) -> list[Path]:
    """Return .safetensors files in a directory, newest first."""
    if not d.exists():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _adapter_dirs() -> dict[str, Path]:
    """Directories likely to contain LoRA adapter checkpoints.

    Mirrors ``_image_dirs``: returns only paths that exist and actually have
    .safetensors files, keyed by a short display name.
    """
    dirs: dict[str, Path] = {}
    for name, path in [
        ("output/ckpt", ROOT / "output" / "ckpt"),
        ("output_temp", ROOT / "output_temp"),
        ("models/diffusion_models", ROOT / "models" / "diffusion_models"),
    ]:
        if path.exists() and any(path.glob("*.safetensors")):
            dirs[name] = path
    # Any subdirectory of output/ckpt/ or output_temp/ with .safetensors (e.g.
    # iteration snapshots). Skip *-checkpoint-state dirs — those are
    # optimizer/state shards, not adapters.
    for parent, label in (
        (ROOT / "output" / "ckpt", "output/ckpt"),
        (ROOT / "output_temp", "output_temp"),
    ):
        if not parent.exists():
            continue
        for p in sorted(parent.iterdir()):
            if (
                p.is_dir()
                and not p.name.endswith("-checkpoint-state")
                and any(p.glob("*.safetensors"))
            ):
                dirs[f"{label}/{p.name}"] = p
    return dirs


def _image_dirs() -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name, path in [
        ("image_dataset", ROOT / "image_dataset"),
        ("post_image_dataset/resized", ROOT / "post_image_dataset" / "resized"),
        ("ip-adapter-dataset", ROOT / "ip-adapter-dataset"),
        ("easycontrol-dataset", ROOT / "easycontrol-dataset"),
        ("output/tests", ROOT / "output" / "tests"),
    ]:
        if path.exists():
            dirs[name] = path
    return dirs


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return w
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        w.setRange(0, 10000)
        w.setValue(v)
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.wheelEvent = lambda e: e.ignore()
        return w
    if isinstance(v, float):
        return QLineEdit(f"{v:g}")
    if isinstance(v, list):
        return QLineEdit(json.dumps(v))
    return QLineEdit(str(v))


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    t = w.text()
    if isinstance(orig, float):
        try:
            return float(t)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(t)
        except (json.JSONDecodeError, ValueError):
            pass
    # Normalize Windows-style backslashes pasted into path/string fields.
    # Forward slashes are valid on every OS Python runs on, and avoid
    # downstream TOML escape errors (e.g. "C:\Users" → \U is not a valid
    # TOML escape).
    if "\\" in t:
        t = t.replace("\\", "/")
    return t


# ── ScaledImageLabel ───────────────────────────────────────────


class ScaledImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._src: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)

    def set_source(self, pm: QPixmap):
        self._src = pm
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._src and not self._src.isNull():
            self.setPixmap(
                self._src.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


# ── Public entry point ─────────────────────────────────────────


def main():
    from gui.app import main as _main

    _main()
