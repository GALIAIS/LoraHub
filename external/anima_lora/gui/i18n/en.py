"""English strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "Training Config",
    "tab_ip_adapter": "IP-Adapter",
    "tab_easycontrol": "EasyControl",
    "tab_postfix": "Postfix",
    "tab_methods": "Methods",
    "tab_images": "Dataset",
    "tab_merge": "Merge",
    "tab_preprocess": "Preprocessing",
    # PreprocessingTab
    "preprocess_intro": (
        "Configure caption shuffling and text-bubble masking, then run each "
        "step on demand. The Training Config tab's Train button auto-runs "
        "preprocess with default settings when no cache exists — this tab "
        "is for tuning and for re-running individual steps."
    ),
    "preprocess_text_caching": "Caching (VAE + text)",
    "preprocess_caption_shuffle_variants": "Shuffle variants per caption (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "Generate N caption variants per image. v0 is the pristine original; "
        "v1..v(N-1) are smart-shuffled and (if tag-dropout > 0) have non-prefix "
        "tags independently dropped. The dataloader picks v0 with 20% probability "
        "and v1..v(N-1) uniformly otherwise when use_shuffled_caption_variants=true. "
        "Set to 0 to cache a single pristine caption only."
    ),
    "preprocess_caption_tag_dropout_rate": "Tag dropout rate (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "Per-tag dropout probability applied to v1..v(N-1). Tags up to and "
        "including the first @artist marker are never dropped. Ignored when "
        "shuffle variants ≤ 0."
    ),
    "preprocess_run_te": "Run caching (VAE + text)",
    "preprocess_masking_sam": "SAM3 masking (text bubbles)",
    "preprocess_masking_mit": "MIT masking (manga text)",
    "preprocess_sam_prompts": "SAM prompts (one per line):",
    "preprocess_sam_prompts_tip": (
        "Text prompts SAM3 looks for. One per line. Defaults to 'speech bubble' "
        "and 'text bubble'."
    ),
    "preprocess_sam_threshold": "SAM threshold (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "Minimum confidence for a SAM3 detection to be kept. Lower = more masks "
        "(may include false positives), higher = stricter. Default 0.5."
    ),
    "preprocess_dilate": "Dilate (px):",
    "preprocess_dilate_tip": (
        "Pixels of dilation applied to the binary mask. Larger values blur "
        "mask edges outward. Default 5. Set to 0 to disable."
    ),
    "preprocess_mit_threshold": "MIT text threshold (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "Confidence threshold for the MIT/ComicTextDetector text segmenter. "
        "Default 0.8."
    ),
    "preprocess_run_mask": "Run masking",
    "preprocess_run_sam_mask": "Run SAM masking",
    "preprocess_run_sam_mask_tip": (
        "Run SAM3 bubble segmentation as part of mask generation. "
        "Uncheck to skip SAM and use only MIT (or whichever other "
        "backends are enabled)."
    ),
    "preprocess_run_mit_mask": "Run MIT masking",
    "preprocess_run_mit_mask_tip": (
        "Run MIT/ComicTextDetector text segmentation as part of mask "
        "generation. Uncheck to skip MIT and use only SAM."
    ),
    "preprocess_mask_nothing_enabled": (
        "At least one of SAM or MIT masking must be enabled."
    ),
    "preprocess_status_resized": "Resized images: {n}",
    "preprocess_status_caches": "Caches — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "Masks: {masks}",
    "preprocess_status_no_resized": "No resized images yet — run Preprocess in the Training Config tab first.",
    "preprocess_log_placeholder": "Preprocessing output will appear here...",
    "preprocess_save_settings": "Save",
    "preprocess_save_settings_tip": "Persist these settings (writes configs/sam_mask.yaml + GUI settings).",
    "preprocess_settings_saved": "Preprocessing settings saved.",
    "preprocess_invalid_float": "Invalid number for {field}: {value}",
    "preprocess_already_running": "A preprocessing step is already running.",
    # ConfigTab
    "preset": "Preset:",
    "save": "Save",
    "save_dirty_tooltip": "Form has unsaved edits. Click Save to write them to the variant file (Train/Preprocess auto-saves first if you skip this).",
    "train": "Train",
    "test": "Test",
    "stop": "Stop",
    "log_placeholder": "Training output will appear here...",
    "from_base": "From base.toml",
    "saved": "Saved",
    "saved_file": "Saved {name}",
    "invalid_toml": "Invalid TOML",
    "error": "Error",
    "accelerate_not_found": "accelerate not found on PATH",
    "preprocess": "Preprocess",
    "preprocess_required": "Please run Preprocess before training.",
    "preprocess_existing_caches_title": "Existing caches will be reused",
    "preprocess_existing_caches_body": (
        "Cache files already exist in:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "Preprocess will REUSE these — they are NOT deleted or "
        "regenerated. Only missing entries will be processed.\n\n"
        "If you want to force a full rebuild (e.g. after editing "
        "captions or changing tokenizer settings), cancel and delete "
        "the cache directory first."
    ),
    "preprocess_cache_count_latents": "{n} VAE latents (.npz)",
    "preprocess_cache_count_te": "{n} text embeddings (_te.safetensors)",
    "preprocess_cache_count_pe": "{n} PE features (_pe.safetensors)",
    "train_using_cache_title": "Use cached dataset?",
    "train_using_cache_body": (
        "A preprocessed dataset cache already exists at:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "Training will reuse this cache as-is. If you've added new images "
        "or edited captions and want them included, cancel and run "
        "Preprocess first.\n\n"
        "Proceed with the existing cache?"
    ),
    "train_autopreprocess_log": (
        "No preprocessed cache found — running preprocess first, "
        "then training automatically.\n"
    ),
    "train_preprocessing": "Preprocessing…",
    "no_lora_for_test": "No LoRA in output/ckpt/ to test. Run training first.",
    "test_output_title": "Latest test output",
    "test_output_empty": "output/tests/ is empty.",
    "finished": "--- Finished (exit code {code}) ---",
    "starting": "Starting… (loading torch / accelerate)",
    "update_success_title": "Update applied",
    "update_success_message": (
        "anima_lora was updated to {v}.\n\n"
        "Close and relaunch the GUI to load the new code."
    ),
    "update_success_badge": "Updated → {v} (relaunch to apply)",
    "update_dryrun_done_title": "Dry run finished",
    "update_dryrun_done_message": (
        "Dry run completed — no files were written. "
        "Review the log to see what a real update would change."
    ),
    "update_failed_title": "Update failed",
    "update_failed_message": (
        "Update exited with code {code}. "
        "See the log for details; the working tree may be partially modified."
    ),
    "resume_checkpoint_title": "Resume training?",
    "resume_checkpoint_question": (
        "A resumable checkpoint was found at step {step}.\n\n"
        "• Yes — resume training from step {step}\n"
        "• No — discard the checkpoint and start fresh\n"
        "• Cancel — don't launch training"
    ),
    "resume_checkpoint_delete_failed": "Could not remove old checkpoint state:\n{error}",
    "locked_by_preset": "Locked by preset (performance settings are fixed for this VRAM profile)",
    "lora_variants": "LoRA Variants",
    "variant": "Variant:",
    "apply_variant": "Apply",
    "apply_variant_tooltip": "Fill the form below with this variant's preset values. Nothing is saved until you click Save.",
    "show_guide": "Guide",
    "show_guide_tooltip": "Show the variant guide and Apply-semantics note in the right panel.",
    "click_field_for_help": "Click a field label to see its explanation here.",
    "no_help_available": "No help available for this field.",
    "extra_args_toggle": "+ Extra args",
    "extra_args_placeholder": "TOML lines for fields not in the form, e.g.\nmy_new_flag = true\nsome_value = 5e-5",
    "extra_args_tooltip": "Add config keys not shown in the form. Parsed as TOML on Save and merged into the current variant file. The form reloads so new keys appear as widgets afterwards. Overrides a form widget if the same key appears in both.",
    "new_variant": "+ New",
    "new_variant_tooltip": "Create a new custom variant under configs/gui-methods/custom/<name>.toml.",
    "new_variant_prompt": "Name for the new variant (saved to configs/gui-methods/custom/<name>.toml).\nLetters, digits, _ and - only.",
    "new_variant_invalid": "Invalid name. Use letters, digits, _, - only.",
    "new_variant_exists": "Variant '{name}' already exists.",
    "basic_section": "Basic",
    "advanced_section": "Advanced (click to expand)",
    # AdapterTab (IP-Adapter / EasyControl)
    "adapter_source_dir": "Source dataset:",
    "adapter_cache_dir": "Cache directory:",
    "adapter_n_pairs": "{n} image / {c} caption pairs",
    "adapter_n_caches": "{n} cached",
    "adapter_preprocess": "Preprocess (resize + VAE + text)",
    "adapter_preprocess_pe": "Preprocess (resize + VAE + text + PE)",
    "adapter_train": "Train",
    "adapter_stop": "Stop",
    "adapter_log_placeholder": "Run output will appear here...",
    "adapter_no_dataset": "Source dataset directory does not exist. Create it and drop in image+caption pairs.",
    "adapter_open_dir": "Open directory",
    "n_images": "{n} images",
    # ImageViewerTab
    "directory": "Directory:",
    "dataset_reload": "Reload",
    "dataset_reload_tooltip": "Re-scan the current directory and refresh the image list and selection.",
    "dataset_add_dir": "Add directory…",
    "dataset_add_dir_tooltip": "Pick another directory and add it to the dropdown for this session.",
    "dataset_add_dir_picker": "Pick a directory to add",
    "dataset_add_dir_already": "Directory '{name}' is already in the list.",
    "dataset_search_placeholder": "Search filename…",
    "dataset_sort_asc_tooltip": "Sort A→Z (click to reverse)",
    "dataset_sort_desc_tooltip": "Sort Z→A (click to reverse)",
    "dataset_mask_overlay": "Show mask overlay",
    "dataset_view_list_tooltip": "Flat list view (click to switch to tree view)",
    "dataset_view_tree_tooltip": "Folder tree view (click to switch to list view)",
    "n_images_filtered": "{shown} / {total} images",
    "caption": "Caption:",
    "no_caption": "(no caption)",
    "caption_save": "Save",
    "caption_revert": "Revert",
    "caption_versions": "Versions…",
    "caption_dirty_marker": " *",
    "caption_diff_stats": "(+{add} / −{rem})",
    "caption_diff_clean": "(no changes)",
    "caption_save_failed": "Failed to save caption: {err}",
    "caption_unsaved_title": "Unsaved caption",
    "caption_unsaved_body": "You have unsaved caption edits. Save before switching?",
    "caption_versions_title": "Caption history — {name}",
    "caption_versions_empty": "(no prior versions)",
    "caption_versions_restore": "Restore selected",
    "caption_versions_close": "Close",
    "caption_no_history": "No history yet for this caption.",
    "caption_guideline_html": (
        "<b>Order:</b> rating → count → character (series) → series → "
        "<span style='color:#c9a227;'>@artist</span> → content tags. "
        "Per-region sub-sections: end the previous tag with <code>.</code> and "
        "start the next with <span style='color:#5e8eb0;'>On the&nbsp;…,</span> "
        "or <span style='color:#5e8eb0;'>In the&nbsp;…,</span>. "
        "Tags up to and including the first <code>@artist</code> are kept fixed; "
        "everything after is shuffled within each section. "
        "<b>No artist?</b> Drop in "
        "<span style='color:#c9a227;'>@no-artist</span> as a placeholder — "
        "it anchors the shuffle boundary the same way and is stripped before "
        "tokenization, so it never reaches the model."
    ),
    # Language
    "language": "Language:",
    # Guidebook
    "guidebook": "📖 Guide",
    "guidebook_tooltip": "Open the Korean end-to-end guide (docs/guidelines/가이드북.md)",
    "guidebook_missing": "Guide not found at {path}",
    "guidebook_open_external": "Open in system viewer",
    "guidebook_close": "Close",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "Models",
    "models_btn_tooltip": "Download or re-download model checkpoints (Anima base, SAM3, MIT, IP-Adapter encoders)",
    "update_btn": "Update",
    "update_btn_tooltip": "Pull the latest anima_lora release from GitHub and run uv sync",
    "report_issue": "Report Issue",
    "report_issue_tooltip": "Open the GitHub issue tracker in your browser",
    "experimental_features": "🧪 Experimental",
    "experimental_features_tooltip": "Open Postfix and IP-Adapter / EasyControl tabs (image-conditioning methods)",
    "experimental_features_title": "Experimental Features",
    # Models dialog
    "models_title": "Download Models",
    "models_intro": "Pick a model group below or use 'Download all' for the standard set "
    "(Anima + SAM3 + MIT + PE). Files are saved under models/.",
    "models_download_all": "Download all (Anima + SAM3 + MIT + PE)",
    "models_download": "Download",
    "models_redownload": "Re-download",
    "models_installed": "✓ Installed",
    "models_missing": "✗ Missing",
    "model_anima": "Anima — DiT + text encoder + VAE",
    "model_sam3": "SAM3 — text-bubble masking",
    "model_mit": "MIT — manga text masking",
    "model_pe": "PE-Core-L14-336 — IP-Adapter vision encoder",
    # Update dialog
    "update_title": "Update anima_lora",
    "update_warning": "Update will pull the latest release from GitHub and overwrite the working "
    "tree (datasets, output/, models/ are preserved). For configs/methods/ "
    "and configs/gui-methods/, choose whether to keep your edits or overwrite "
    "them with upstream (your version is backed up first). Run 'Dry run' to "
    "preview the changes.",
    "update_dry_run": "Dry run",
    "update_run": "Run update",
    "update_run_keep": "Update — keep my configs",
    "update_run_overwrite": "Update — overwrite configs (back up mine)",
    "update_confirm": "This will rewrite anima_lora source files. Continue?",
    "update_check_now": "Check now",
    "update_view_release": "View on GitHub",
    "update_current_version": "Current: {v}",
    "update_latest_version": "Latest: {v}",
    "update_no_baseline": "unknown (no manifest)",
    "update_status_checking": "Checking…",
    "update_status_uptodate": "✓ Up to date",
    "update_status_available": "● Update available",
    "update_status_unknown": "? Cannot compare (no local manifest)",
    "update_status_failed": "✗ Check failed",
    "update_release_notes": "Release notes:",
    "update_no_release_notes": "(release has no description)",
    "update_check_error": "Could not reach GitHub: {err}",
    # MergeTab
    "n_files": "{n} files",
    "merge_no_adapter": "No adapters found",
    "merge_no_adapter_msg": "No adapter selected or the file doesn't exist.",
    "merge_no_selection": "Select a checkpoint from the list to scan it.",
    "merge_verdict_ready": "✓ Ready to bake",
    "merge_verdict_partial": "△ Partial — LoRA bakeable, ReFT will be dropped",
    "merge_verdict_hydra": "✗ HydraLoRA moe — layer-local router can't be baked",
    "merge_verdict_postfix_only": "✗ Postfix/prefix only — not a weight delta",
    "merge_verdict_reft_only": "✗ ReFT only — block-level hook, no LoRA to bake",
    "merge_verdict_unknown": "? No recognized adapter keys",
    "merge_options": "Merge Options",
    "merge_base_dit": "Base DiT:",
    "merge_multiplier": "Multiplier:",
    "merge_multiplier_tip": "LoRA strength to bake in (1.0 = full strength).",
    "merge_dtype": "Save dtype:",
    "merge_out": "Output:",
    "merge_out_placeholder": "(auto: <adapter>_merged.safetensors)",
    "merge_allow_partial": "Allow partial merge (drop ReFT / Hydra / postfix keys)",
    "merge_allow_partial_tip": "Proceed even if the adapter contains non-bakeable components. Dropped components will be absent from the merged DiT.",
    "merge_button": "Merge into DiT",
    "merge_log_placeholder": "Merge output will appear here...",
    "merge_pick_dir": "Select adapter directory",
    "merge_pick_file": "Select adapter .safetensors",
    "merge_pick_dit": "Select base DiT .safetensors",
    "merge_pick_out": "Save merged DiT as...",
    "browse": "Browse…",
}
