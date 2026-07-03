# LoraHub Review Guide

This file defines how Codex should review changes in this repository. Treat it as a code-review rubric, not a development plan.

## Product Contract

LoraHub is a Windows/Linux LoRA training workbench. It manages long-running GPU jobs, local model files, datasets, generated artifacts, backend installers, and a React control surface. A correct change is not only one that compiles. It must keep user data safe, keep jobs controllable, and leave the app recoverable after failure.

Repository map:

| Area | Path |
| --- | --- |
| Python backend and CLI | `lorahub/` |
| FastAPI routes | `lorahub/api/` |
| React/Vite frontend | `web/` |
| Training backends | `external/` |
| Tests | `tests/` |
| User-owned data | `configs/`, `runs/`, `models/`, `output/`, `datasets/`, `.env*` |

## Review Rules

### 1. Prove the change is in the right layer

- Prefer one root-cause fix in shared code over scattered guards in call sites.
- Do not accept UI-only fixes for backend state bugs, or backend-only fixes for stale frontend cache bugs.
- If a change crosses API, scheduler, config compiler, backend launcher, and UI, verify the contract at every boundary.
- Reject speculative abstractions, parallel frameworks, or new dependencies unless the existing code cannot reasonably solve the issue.

### 2. Preserve user data by default

- Do not approve any path that can delete, overwrite, move, or clean user-owned data without explicit user opt-in.
- Protected paths include `configs/`, `runs/`, `models/`, `output/`, `datasets/`, `.env*`, `external/anima_lora/output/`, and `external/anima_lora/post_image_dataset/`.
- Update, upgrade, install, rebuild, and force modes must preserve user config and restore it after failure.
- Archive extraction, downloads, imports, cleanup, and sync operations must defend against path traversal and symlink escape.

### 3. Keep job lifecycle deterministic

- Training, captioning, tagging, downloads, imports, previews, and backend installs must have clear states: queued, starting, running, stopping, stopped, failed, completed.
- Cancel and force-stop must work before and during preprocessing, not only after the main training loop starts.
- Do not report success until child processes are healthy and required files exist.
- Avoid orphaned subprocesses, stale lock files, duplicate queue entries, and mismatched UI/backend state.
- Progress, loss, validation, preview images, and errors must survive page refresh when the backend task is still running.

### 4. Treat subprocess and filesystem boundaries as security boundaries

- Use structured argument arrays for subprocesses. Avoid shell strings.
- Validate user-controlled paths before filesystem access, model loading, archive creation, extraction, or process launch.
- Never log secrets: tokens, cookies, SSH keys, Hugging Face tokens, ModelScope tokens, proxy credentials, or `.env` contents.
- Terminal or unrestricted command features must have explicit routing, auditability, and clear user intent.

### 5. Keep backend schemas and generated configs exact

- Public config fields must remain backward-compatible unless a migration is included.
- Generated configs must match the target backend schema exactly. Do not leak UI sentinel values, booleans, labels, or empty strings into fields that require real paths or strings.
- For `anima_lora`, review precision, sampler, validation, sampling, optimizer, network algorithm, checkpoint cadence, LoRA/full-finetune mode, and distributed launch arguments against the actual command line.
- For `ai_toolkit`, review generated YAML, model architecture fields, sample prompt fields, LoRA network settings, preview behavior, and dependency assumptions against ai-toolkit runtime expectations.

### 6. Make GPU behavior explicit

- GPU dispatch must match the user's selected mode: single GPU per task, concurrent tasks across slots, or one task across multiple GPUs.
- Verify `CUDA_VISIBLE_DEVICES`, accelerate/deepspeed/fsdp settings, process count, UI labels, and status bar data agree.
- Mixed GPU systems must not silently choose an unsafe distributed mode.
- Precision changes must account for hardware limits, especially V100 fp16 stability and fp32 fallback behavior.

### 7. Design frontend changes for real workflows

- Dense app pages must remain scannable, responsive, keyboard-accessible, and usable in light mode, dark mode, desktop, and mobile.
- Every changed flow needs usable loading, empty, error, disabled, retry, and long-content states when applicable.
- Do not approve decorative UI that hides controls, removes local scrolling, makes logs harder to read, or increases vertical bloat without adding workflow value.
- Theme/style changes must preserve contrast, focus states, chart readability, and shadcn-compatible token behavior.

### 8. Keep installer and updater flows recoverable

- Review Windows `.bat`/PowerShell and Linux shell paths separately. Quoting that works on one platform may fail on the other.
- Network installers must use mirror fallback, actionable logs, timeouts, and retry behavior where downloads are large or unreliable.
- Updating must rebuild, restart on the prior port, and surface useful logs without leaving the service half-updated.
- Builds must tolerate locked `web/dist` files on Windows and avoid silent stale frontend serving.

### 9. Make logs and error reports useful, not noisy

- Log streams must be append-safe, resumable, and efficient for large outputs.
- One traceback should become one actionable error report, not many fragmented reports.
- Error reports must include task id, workspace, command context, return code, and relevant log tail when safe.
- Do not suppress backend errors just to keep UI status green.

### 10. Require risk-matched verification

- Small UI token or copy changes: build or targeted check is enough.
- Backend scheduler, config, installer, updater, or filesystem changes: require targeted tests or a concrete manual verification note.
- Training backend precision, distributed, or algorithm changes: require a smoke launch or a clear explanation of why it cannot be run.
- If verification is skipped, state the remaining risk directly.

## Review Output Format

Lead with findings. Do not start with a summary.

Each finding must include:

| Field | Requirement |
| --- | --- |
| Severity | `blocker`, `high`, `medium`, or `low` |
| Location | File and line |
| Failure mode | What breaks |
| User impact | Why it matters |
| Fix direction | Minimal correction |

If no issues are found, say that clearly and list any tests that were not run.

## Checks

Use the smallest relevant set:

| Change area | Check |
| --- | --- |
| Python targeted behavior | `python -m pytest tests/test_name.py` |
| Python broad backend behavior | `python -m pytest tests` |
| Python lint | `python -m ruff check .` |
| Python types | `python -m mypy lorahub` |
| Frontend build | `npm --prefix web run build` |
| Frontend visual/interaction | Browser verification for the touched page |

Do not demand unrelated full-suite runs when a targeted test covers the risk.

## Merge Bar

Do not approve a PR that:

- can lose, delete, overwrite, or hide user data
- breaks install, update, build, or service restart on Windows or Linux
- makes a running job impossible to stop or recover
- changes training launch behavior without matching config/compiler verification
- weakens path, archive, download, terminal, or subprocess safety
- makes logs, progress, loss, validation, previews, or errors less reliable
- regresses frontend contrast, focus, mobile layout, dark mode, or light mode
- introduces broad abstractions or dependencies without a concrete need

Approve only when the diff is scoped, the failure modes are accounted for, and the verification is proportional to the risk.
