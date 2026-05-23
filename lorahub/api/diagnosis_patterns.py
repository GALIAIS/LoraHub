"""Failure-mode regex catalogue for ``diagnose_failure``.

Each entry is `(category, pattern, severity, message, remediation)`:

- **category**: short slug, used to dedupe (one finding per category per
  run). Pick stable identifiers — they leak into the UI's
  diagnostic_warning event payload.
- **pattern**: regex (case-insensitive at match time) tested against a
  combined haystack of stderr, the tail of training.log, and the most
  recent events.jsonl entries.
- **severity**: ``error`` for things that explain a non-zero exit;
  ``warn`` for things that don't kill the run but the user should know;
  ``info`` for benign markers (user cancel).
- **message**: one-line user-facing summary in plain English.
- **remediation**: actionable next step, with concrete CLI flags / file
  paths / cfg field names where possible.

Adding a new entry: prefer specific, low-false-positive patterns. A
regex that fires on every run with a torch warning is worse than no
regex — it trains users to ignore the panel.
"""

from __future__ import annotations

from typing import Literal

Severity = Literal["info", "warn", "error"]


_PATTERNS: list[tuple[str, str, Severity, str, str]] = [
    # --------------------------- existing rules ----------------------- #
    (
        "oom",
        r"(CUDA out of memory|OutOfMemoryError|cudaMalloc.*failed|memory error)",
        "error",
        "GPU ran out of memory mid-training.",
        "Reduce --train_batch_size, raise --gradient_accumulation_steps, "
        "enable --gradient_checkpointing, or pick a higher --blocks_to_swap "
        "value (anima only).",
    ),
    (
        "nan_loss",
        # Match the *narrative* a trainer prints when loss diverges, not the
        # CLI flags that *configure* nan handling. Anchoring with word
        # boundaries + a negative lookbehind/lookahead for argv noise (`-`,
        # `_`, `'`) keeps a CalledProcessError repr that lists --nan_guard
        # and --masked_loss in the same line from triggering this rule via
        # the old greedy `NaN.*loss`. Real trainers print one of:
        #   * "non-finite loss at global_step=…"           (anima_lora loop.py)
        #   * "Loss is NaN, skipping update"               (sd-scripts)
        #   * "loss became NaN"                            (kohya)
        # All three forms stay covered.
        r"(?:"
        r"\bnon-finite\s+loss\b"
        r"|(?<![\w'\"-])loss\s+(?:is|became)\s+nan\b"
        r"|(?<![\w'\"-])\bnan\s+loss\b(?!_)"
        r")",
        "error",
        "Loss became NaN — training is numerically unstable.",
        "Try --nan_guard --nan_guard_recover to skip bad steps + auto-recover. "
        "If it keeps firing, lower --learning_rate by 2x or switch to fp32 "
        "mixed precision.",
    ),
    (
        "missing_module",
        r"ModuleNotFoundError: No module named ['\"](\w+)['\"]",
        "error",
        "A Python dependency is missing.",
        "Re-run scripts/install.sh (or scripts/install-cn.sh inside China). "
        "If the missing package is bitsandbytes/lpips/came-pytorch, those "
        "are optional — install with: uv pip install <package>.",
    ),
    (
        "missing_safetensors",
        r"(FileNotFoundError|No such file or directory).*\.(safetensors|ckpt|pt)",
        "error",
        "A model file (safetensors / checkpoint) wasn't found.",
        "Run `make download-models` (anima) or `lorahub doctor` to see "
        "which model paths are missing. Check your YAML's model_paths.",
    ),
    (
        "torch_compile_fail",
        r"(torch._dynamo|InductorError|fx.GraphModule.*failed|recompile)",
        "warn",
        "torch.compile fell back to eager (training continues but slower).",
        "Set --no_compile or check that your CUDA/PyTorch versions match. "
        "Compile failures are usually harmless; only worry if step time "
        "doubled.",
    ),
    (
        "data_loader_corrupt",
        r"(corrupt.*image|cannot identify image file|truncated.*image|"
        r"PIL.UnidentifiedImageError)",
        "error",
        "An image in the dataset is corrupt or unreadable.",
        "Run scripts/validate_dataset.py or open --dataset_dir manually "
        "and remove the offending file. The traceback above usually "
        "names the file path.",
    ),
    (
        "user_cancel",
        r"(KeyboardInterrupt|received signal|user aborted)",
        "info",
        "Training was canceled by the user (Ctrl+C / API cancel).",
        "Resume with the same recipe — the saver wrote a checkpoint at "
        "the last save_every_n_epochs boundary.",
    ),
    (
        "vram_pressure",
        r"(RuntimeError.*CUDA error.*out of memory)",
        "error",
        "Driver reported OOM during a CUDA call.",
        "Same as oom: reduce batch / accumulate / checkpoint. If this "
        "happens at startup, another process is hogging VRAM — check "
        "nvidia-smi.",
    ),

    # --------------------------- new rules ---------------------------- #
    (
        "ansi_encode",
        # Trips on the manage-install class of bug AND on sd-scripts
        # passing CJK paths through Windows mbcs.
        r"UnicodeEncodeError.*'(?:ascii|cp\d+|gbk|charmap|mbcs)'.*can't encode",
        "error",
        "A path or string with non-ASCII characters reached a Windows-only "
        "ANSI/ASCII boundary.",
        "Move the project, dataset, or model out of any directory whose "
        "name contains characters not representable in the active Windows "
        "ANSI code page (emoji, mixed-script names, accented letters on a "
        "GBK box). The preflight check `path_encoding` lists every cfg "
        "field that looked unsafe.",
    ),
    (
        "cjk_path_decode",
        r"UnicodeDecodeError.*(?:gbk|cp936|cp1252|charmap).*can't decode",
        "error",
        "cmd.exe / a child process emitted bytes that Python could not "
        "decode under the active ANSI code page.",
        "Same fix as ansi_encode: keep the entire training-relevant path "
        "tree in characters representable by your Windows ANSI code page. "
        "The encoding mismatch usually originates in subprocess output, "
        "not the trainer itself.",
    ),
    (
        "cuda_driver_mismatch",
        r"(CUDA driver version is insufficient|CUDA error.*no kernel image is available|"
        r"forward compatibility was attempted on non supported HW)",
        "error",
        "Installed CUDA driver doesn't match the PyTorch / accelerate build.",
        "Update the NVIDIA driver to a version compatible with the CUDA "
        "runtime PyTorch was built against (e.g. CUDA 12.x needs driver "
        ">=525 on Linux, >=528 on Windows). `nvidia-smi` shows the "
        "installed driver version.",
    ),
    (
        "accelerate_config_missing",
        r"(accelerate config not found|default config does not exist|"
        r"`accelerate config` first|run accelerate config to configure)",
        "error",
        "accelerate has no configuration file for the current user.",
        "Run `accelerate config default` once, accept the prompts, then "
        "retry. The config lives at ~/.cache/huggingface/accelerate/"
        "default_config.yaml.",
    ),
    (
        "permission_denied_write",
        # Differentiate from PermissionError on read by requiring "write" /
        # "create" / WinError 5 in the same line.
        r"(PermissionError.*Permission denied|\[WinError 5\]|"
        r"OSError.*Operation not permitted)",
        "error",
        "Filesystem refused a write — the trainer cannot save the "
        "checkpoint or write a log.",
        "Check the workspace and output directories: another process may "
        "hold an open handle, antivirus may be quarantining .safetensors, "
        "or the path may live under a directory the user lacks write "
        "permission on.",
    ),
    (
        "disk_full",
        r"(No space left on device|\[Errno 28\]|"
        r"There is not enough space on the disk|"
        r"OSError.*disk full)",
        "error",
        "Workspace partition ran out of free space mid-write.",
        "Free up space on the workspace volume (state directories and "
        ".safetensors are large). Set resume.saveLastNEpochsState to keep "
        "only N most recent state dirs, or move output.outputDir to a "
        "roomier disk and re-launch.",
    ),
    (
        "bitsandbytes_missing",
        r"(bitsandbytes.*not.*installed|No module named ['\"]bitsandbytes['\"]|"
        r"compiled without 8-bit optimizer)",
        "error",
        "8-bit optimizer requested but bitsandbytes isn't installed in the "
        "active venv.",
        "Either install it (`uv pip install bitsandbytes`) or change "
        "optimizer.optimizerType away from a 8bit optimizer (AdamW8bit, "
        "Lion8bit, AdaFactor8bit etc.).",
    ),
    (
        "xformers_incompat",
        r"(xformers.*incompat|xFormers wasn't built with CUDA|"
        r"xformers.*has no attribute)",
        "warn",
        "xformers is installed but doesn't match the active torch / CUDA build.",
        "Either reinstall xformers from the wheel matching your torch "
        "version (https://github.com/facebookresearch/xformers/wiki/"
        "Installing-xformers) or disable it via the backend's "
        "no_xformers / use_flash_attn flag.",
    ),
    (
        "safetensors_corrupt",
        r"(SafetensorError|safetensors_rust.SafetensorError|"
        r"InvalidHeaderDeserialization|invalid SafeTensors header)",
        "error",
        "A .safetensors file failed to deserialize — the file is "
        "incomplete or corrupt.",
        "Re-download the model file (a partial download is the usual "
        "cause). For a checkpoint produced by a prior run, the run was "
        "killed during save — delete the half-written .safetensors and "
        "resume from an earlier state.",
    ),
    (
        "caption_missing",
        r"(No caption file for|caption.*missing for|"
        r"could not find caption.*for image)",
        "error",
        "A dataset image has no matching caption.txt file.",
        "Either generate captions for the offending images (image-studio "
        "→ AI 标注), set dataset.caption.required=false in the recipe, "
        "or remove the orphan images.",
    ),
    (
        "vram_startup",
        r"(CUDA error.*out of memory.*\b(?:init|create|alloc|cudnn)\b|"
        r"Could not load library.*cuda)",
        "error",
        "GPU rejected the allocation before training even started.",
        "Another process owns the VRAM (kill stray python.exe / "
        "pytorch.exe via Task Manager or `nvidia-smi -kill`). On a single-"
        "GPU box, also close browser tabs running webgl content — they "
        "claim 200-500 MiB.",
    ),
    (
        "deepspeed_nccl",
        r"(NCCL.*error|distributed_c10d.*error|"
        r"NCCL communicator was aborted|all_reduce.*failed)",
        "error",
        "DeepSpeed / NCCL communication call failed.",
        "Check that all participating GPUs have the same driver version "
        "and that NCCL_P2P_DISABLE / NCCL_IB_DISABLE aren't required on "
        "your interconnect. Single-node multi-GPU on Windows often needs "
        "NCCL_SOCKET_IFNAME set explicitly.",
    ),
    (
        "distributed_timeout",
        r"(torch\.distributed.*timeout|Watchdog caught collective operation timeout|"
        r"DDP timeout|process group has not been initialized)",
        "error",
        "Multi-rank training stalled past the collective-op timeout.",
        "One rank is much slower than the others (different GPU model, "
        "different num_workers) or stuck on disk IO. Lower the dataset's "
        "num_workers, equalise the GPUs, or raise the NCCL timeout via "
        "the NCCL_BLOCKING_WAIT / TORCH_NCCL_TIMEOUT_MS env vars.",
    ),
    (
        "subprocess_returncode",
        r"(subprocess.CalledProcessError|returned non-zero exit status|"
        r"command exited with code [1-9])",
        "warn",
        "A trainer-spawned subprocess exited non-zero.",
        "The matched line usually names the failing command (accelerate "
        "launch / git / curl / pip). Open training.log around the match "
        "to see the subprocess's own stderr.",
    ),
]


def get_patterns() -> list[tuple[str, str, Severity, str, str]]:
    """Return the catalogue. Returned list is shared; callers must not mutate."""
    return _PATTERNS


__all__ = ["get_patterns", "Severity"]
