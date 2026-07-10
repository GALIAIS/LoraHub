"""Centralised CLI strings for the user-facing front end of lorahub.

Keep API / library code in English (per the project guideline at
`CLAUDE.md`); the CLI layer is the boundary where we translate to
the user's preferred language. Default is Simplified Chinese
because the workbench's primary audience writes Chinese; English
is opt-in via ``lorahub --lang en …`` or ``LORAHUB_LANG=en``.

The dictionary uses English keys so a missing translation falls
back to a still-readable string instead of an opaque token.

This module deliberately does not depend on Typer or Rich — call
sites import ``t`` and pass the rendered string into whichever
console / typer.Option / typer.Exit context they need.
"""

from __future__ import annotations

import os
from typing import Final, Literal

Lang = Literal["zh", "en"]

_DEFAULT_LANG: Lang = "zh"
_VALID_LANGS: Final[frozenset[str]] = frozenset({"zh", "en"})

# Mutable per-process language. Set once by the typer global callback
# in ``cli.main`` before any subcommand body runs; consumed by ``t``.
_current_lang: Lang = _DEFAULT_LANG


def set_lang(lang: str | None) -> Lang:
    """Pin the CLI display language for the rest of this process.

    Resolution order: explicit argument → ``LORAHUB_LANG`` env var →
    default. Unknown values fall back to the default rather than
    raising — getting an English help screen because of a typo is
    less hostile than a stack trace.
    """
    global _current_lang  # noqa: PLW0603
    candidate = (lang or os.environ.get("LORAHUB_LANG") or _DEFAULT_LANG).lower()
    if candidate not in _VALID_LANGS:
        candidate = _DEFAULT_LANG
    _current_lang = candidate  # type: ignore[assignment]
    return _current_lang


def current_lang() -> Lang:
    return _current_lang


# --------------------------------------------------------------------------- #
# Dictionary
#
# Keys are dot-notated to keep navigation easy in your editor. New
# strings should land here first — call sites pull them via ``t(...)``.
# --------------------------------------------------------------------------- #
MESSAGES: dict[str, dict[Lang, str]] = {
    # ── Top-level app help ─────────────────────────────────────────
    "app.help": {
        "zh": "面向扩散模型的开源 LoRA 训练工作台。",
        "en": "Open-source LoRA training workbench for diffusion models.",
    },
    "app.lang.option": {
        "zh": "界面语言:zh / en(默认 zh,可用 LORAHUB_LANG 环境变量覆盖)。",
        "en": "Interface language: zh / en (default zh; override via LORAHUB_LANG).",
    },
    "app.no_tui.option": {
        "zh": "禁用无参数交互式终端界面,输出普通 CLI 帮助。",
        "en": "Disable the no-argument terminal UI and print normal CLI help.",
    },
    "app.tui.option": {
        "zh": "强制打开交互式终端界面。",
        "en": "Force the interactive terminal UI.",
    },
    "app.version.help": {
        "zh": "打印已安装的 lorahub 版本号。",
        "en": "Print the installed lorahub version.",
    },
    "app.doctor.help": {
        "zh": "自检本地安装(venv / Python / Node / web/dist / 后端)。",
        "en": "Inspect the local install — venv, Python, Node, web/dist, backends.",
    },
    # ── manage (formerly self) sub-app ─────────────────────────────
    "manage.help": {
        "zh": "管理 lorahub 自身的安装(更新 / 升级 / 重建 / PATH 注册)。",
        "en": "Manage the lorahub install itself (update / upgrade / build / PATH).",
    },
    "manage.path.help": {
        "zh": "打印当前使用的 `lorahub` 命令所在路径。",
        "en": "Print the location of the currently-running ``lorahub`` command.",
    },
    "manage.path.shutil_which": {"zh": "shutil.which:   ", "en": "shutil.which:   "},
    "manage.path.venv_entry": {"zh": "venv 入口:     ", "en": "venv entry:     "},
    "manage.path.shim": {"zh": "用户 PATH 启动器:", "en": "user-PATH launcher:"},
    "manage.path.not_on_path": {"zh": "不在 PATH 上", "en": "not on PATH"},
    "manage.path.none": {"zh": "未安装", "en": "none"},
    "manage.path.exists": {"zh": "已存在", "en": "exists"},
    "manage.path.absent": {"zh": "缺失", "en": "absent"},

    "manage.install.help": {
        "zh": "把 `lorahub` 启动器写入用户 PATH。",
        "en": "Add a ``lorahub`` launcher to the user PATH.",
    },
    "manage.install.no_venv_entry": {
        "zh": "[red]当前 venv 中没有 Python 解释器[/]\n请先运行 scripts/install.{sh,bat} 创建 .venv。",
        "en": "[red]no Python interpreter in the active venv[/]\nRun scripts/install.{sh,bat} first to create .venv.",
    },
    "manage.install.path_unencodable": {
        "zh": "[red]无法写入 lorahub 启动器:venv 入口路径含有当前 Windows ANSI 代码页(mbcs)无法编码的字符。[/]\n  venv 入口:{venv_entry}\n  编码错误:{err}\n请把项目移到只包含 ANSI 代码页可表示字符的目录(避免中文以外的特殊符号、表情或当前代码页未覆盖的字符)。",
        "en": "[red]cannot write lorahub launcher: the venv entry path contains characters the active Windows ANSI code page (mbcs) cannot encode.[/]\n  venv entry: {venv_entry}\n  encode error: {err}\nMove the project to a directory whose path only uses characters representable in your current ANSI code page.",
    },
    "manage.install.setx_failed": {
        "zh": "[yellow]启动器已写入 {shim},但 setx PATH 失败:[/] {err}\n请手动把 {shim_dir} 加到用户 PATH。",
        "en": "[yellow]wrote launcher {shim}, but setx PATH failed:[/] {err}\nAdd this to your user PATH manually: {shim_dir}",
    },
    "manage.install.windows_done": {
        "zh": "[green]已安装[/] {shim}\n[dim]已把 {shim_dir} 加到用户 PATH(打开新终端后生效)[/]",
        "en": "[green]installed[/] {shim}\n[dim]added {shim_dir} to user PATH (open a new shell to use it)[/]",
    },
    "manage.install.posix_done": {
        "zh": "[green]已安装[/] {shim} -> {target}",
        "en": "[green]installed[/] {shim} -> {target}",
    },
    "manage.install.path_hint": {
        "zh": "[yellow]提示:[/] {shim_dir} 不在 PATH 上。\n把它加到 shell 启动文件,例如:\n  echo 'export PATH=\"{shim_dir}:$PATH\"' >> ~/.bashrc",
        "en": "[yellow]note:[/] {shim_dir} is not on your PATH.\nAdd it to your shell rc, e.g.:\n  echo 'export PATH=\"{shim_dir}:$PATH\"' >> ~/.bashrc",
    },

    "manage.uninstall.help": {
        "zh": "从用户 PATH 中移除 `lorahub` 启动器。",
        "en": "Remove the ``lorahub`` launcher from the user PATH.",
    },
    "manage.uninstall.no_shim": {
        "zh": "[dim]{shim} 处没有启动器[/]",
        "en": "[dim]no launcher at {shim}[/]",
    },
    "manage.uninstall.removed": {
        "zh": "[green]已移除[/] {shim}",
        "en": "[green]removed[/] {shim}",
    },
    "manage.uninstall.dir_hint": {
        "zh": "[dim]提示:启动器所在目录仍在你的用户 PATH 上。[/]\n如想彻底清理,请通过 设置 → 环境变量 移除。",
        "en": "[dim]note: the launcher directory is still on your user PATH.[/]\nRemove it via Settings → Environment Variables if you want a clean slate.",
    },

    "manage.update.help": {
        "zh": "拉取 origin/dev 最新代码、重装 Python 依赖、重建前端。",
        "en": "Pull the latest commits from origin/dev, reinstall deps, rebuild SPA.",
    },
    "manage.update.skip_build_help": {
        "zh": "跳过前端重建(更快;只更新后端)。",
        "en": "Skip the frontend rebuild (faster; backend-only update).",
    },
    "manage.update.force_help": {
        "zh": "在拉取前丢弃本地修改(git reset --hard + clean -fd)。等同于设置页里「忽略冲突」开关。",
        "en": "Discard local changes (git reset --hard + clean -fd) before pulling. Same as the Settings UI's 'ignore conflicts' toggle.",
    },
    "manage.update.failed": {
        "zh": "[red]更新失败:[/] {err}",
        "en": "[red]update failed:[/] {err}",
    },
    "manage.update.force_hint": {
        "zh": "[yellow]提示:[/] 加 [bold]--force[/] 重试,可丢弃挡住合并的本地修改。",
        "en": "[yellow]Hint:[/] retry with [bold]--force[/] to discard local changes that block the merge.",
    },
    "manage.update.complete": {"zh": "[green]更新完成[/]", "en": "[green]update complete[/]"},
    "manage.upgrade.help": {
        "zh": "切换工作树到最新发布的 v* tag。",
        "en": "Switch the working tree to the latest published release tag.",
    },
    "manage.upgrade.skip_build_help": {
        "zh": "切换后跳过前端重建。",
        "en": "Skip the SPA rebuild after checkout.",
    },
    "manage.upgrade.force_help": {
        "zh": "在 checkout tag 前丢弃本地修改(git reset --hard + clean -fd)。",
        "en": "Discard local changes (git reset --hard + clean -fd) before checking out the latest tag.",
    },
    "manage.upgrade.failed": {
        "zh": "[red]升级失败:[/] {err}",
        "en": "[red]upgrade failed:[/] {err}",
    },
    "manage.upgrade.force_hint": {
        "zh": "[yellow]提示:[/] 加 [bold]--force[/] 重试,可丢弃挡住合并的本地修改。",
        "en": "[yellow]Hint:[/] retry with [bold]--force[/] to discard local changes.",
    },
    "manage.upgrade.complete": {
        "zh": "[green]升级完成[/]",
        "en": "[green]upgrade complete[/]",
    },
    "manage.restart_hint": {
        "zh": "请重启服务:[bold]lorahub service restart[/]",
        "en": "Restart the daemon: [bold]lorahub service restart[/]",
    },

    "manage.build.help": {
        "zh": "重建前端(vite build),等同于更新流程的最后一步。",
        "en": "Rebuild the web frontend (vite build) — equivalent to update step 3/3.",
    },
    "manage.build.no_npm": {
        "zh": "[red]找不到 npm。[/]请先运行 scripts/install.{sh,bat} 安装便携 Node 工具链。",
        "en": "[red]npm not found.[/] Run scripts/install.{sh,bat} first to install the portable Node toolchain.",
    },
    "manage.build.npm_failed": {
        "zh": "[red]npm run build 失败。[/]",
        "en": "[red]npm run build failed.[/]",
    },
    "manage.build.start": {
        "zh": "构建前端 v{version}",
        "en": "building web frontend v{version}",
    },
    "manage.build.complete": {
        "zh": "[green]构建完成[/]",
        "en": "[green]build complete[/]",
    },

    # Phase prefixes for the streaming progress emit (git / deps / build / done).
    "manage.phase.git": {"zh": "[blue]git[/]", "en": "[blue]git[/]"},
    "manage.phase.deps": {"zh": "[cyan]依赖[/]", "en": "[cyan]deps[/]"},
    "manage.phase.build": {"zh": "[magenta]构建[/]", "en": "[magenta]build[/]"},
    "manage.phase.done": {"zh": "[green]完成[/]", "en": "[green]done[/]"},

    "manage.install.path_too_long": {
        "zh": "[red]新 PATH 长度 {length} 超过 setx 1024 字符上限,继续将截断你的用户 PATH。[/]\n"
              "请先精简用户 PATH(在 设置 → 环境变量),再重试 `lorahub manage install`,"
              "或手动把 {shim_dir} 加进去。",
        "en": "[red]new PATH length {length} exceeds setx's 1024-char ceiling; "
              "proceeding would truncate your user PATH.[/]\n"
              "Trim the user PATH first (Settings → Environment Variables) and "
              "retry `lorahub manage install`, or add {shim_dir} manually.",
    },

    # ── doctor ─────────────────────────────────────────────────────
    "doctor.title": {"zh": "lorahub 自检 — {root}", "en": "lorahub doctor — {root}"},
    "doctor.col.component": {"zh": "组件", "en": "component"},
    "doctor.col.location": {"zh": "路径", "en": "location"},
    "doctor.col.status": {"zh": "状态", "en": "status"},
    "doctor.status.ok": {"zh": "[green]正常[/]", "en": "[green]OK[/]"},
    "doctor.status.missing": {"zh": "[red]缺失[/]", "en": "[red]missing[/]"},
    "doctor.status.warn": {"zh": "[yellow]警告[/]", "en": "[yellow]warn[/]"},
    "doctor.hint.install": {"zh": "请运行 scripts/install.{{sh,bat}}", "en": "run scripts/install.{{sh,bat}}"},
    "doctor.hint.build": {"zh": "请运行 `lorahub manage build`", "en": "run `lorahub manage build`"},
    "doctor.env.title": {"zh": "环境检查", "en": "environment"},
    "doctor.env.col.check": {"zh": "项目", "en": "check"},
    "doctor.env.col.detail": {"zh": "详情", "en": "detail"},
    "doctor.env.path_encoding": {"zh": "ANSI 路径编码", "en": "ANSI path encoding"},
    "doctor.env.path_offending": {
        "zh": "{path}(无法编码区段 {start}-{end})",
        "en": "{path} (offending {start}-{end})",
    },
    "doctor.env.path_hint": {
        "zh": "把项目移到不含特殊字符的目录(emoji / 当前代码页未覆盖的字符)",
        "en": "move the project to a directory without exotic chars (emoji / out-of-codepage)",
    },
    "doctor.env.disk": {"zh": "磁盘空间", "en": "disk space"},
    "doctor.env.disk_hint": {
        "zh": "训练 state 目录单次可达数 GiB,建议 ≥5 GiB",
        "en": "training state can grow several GiB per run; aim for ≥5 GiB",
    },
    "doctor.env.gpu": {"zh": "GPU 可见 (nvidia-smi)", "en": "GPU visible (nvidia-smi)"},
    "doctor.env.gpu_count": {"zh": "{n} 张 GPU", "en": "{n} GPU(s)"},
    "doctor.env.gpu_none": {"zh": "未检测到 GPU", "en": "no GPUs reported"},
    "doctor.env.gpu_hint_cpu": {
        "zh": "若计划在 CPU 训练可忽略;否则检查驱动 / nvidia-smi 是否在 PATH",
        "en": "ignore if you plan CPU-only training; else check driver / nvidia-smi on PATH",
    },
    "doctor.env.gpu_missing": {"zh": "PATH 上未找到 nvidia-smi", "en": "nvidia-smi not found on PATH"},
    "doctor.env.gpu_missing_hint": {
        "zh": "若计划在 GPU 训练,先安装 NVIDIA driver 并把 nvidia-smi 放进 PATH",
        "en": "to train on GPU, install the NVIDIA driver and add nvidia-smi to PATH",
    },
    "doctor.backends.title": {"zh": "训练后端", "en": "backends"},
    "doctor.backends.col.id": {"zh": "id", "en": "id"},
    "doctor.backends.col.ready": {"zh": "就绪", "en": "ready"},
    "doctor.backends.col.python": {"zh": "Python", "en": "python"},
    "doctor.backends.col.notes": {"zh": "备注", "en": "notes"},
    "doctor.backends.ready_yes": {"zh": "[green]是[/]", "en": "[green]yes[/]"},
    "doctor.backends.ready_no": {"zh": "[yellow]否[/]", "en": "[yellow]no[/]"},
    "doctor.backends.note_missing_scripts": {"zh": "缺脚本: {n}", "en": "missing scripts: {n}"},
    "doctor.backends.note_missing_models": {"zh": "缺模型: {n}", "en": "missing models: {n}"},
    "doctor.backends.note_no_venv": {"zh": "无 venv", "en": "no venv"},

    # ── validate / info / train top-level commands ─────────────────
    "cli.config_arg_help": {"zh": "config YAML 文件路径。", "en": "Path to a config YAML file."},
    "cli.workspace_help": {"zh": "日志 / 检查点 / 样本输出目录。", "en": "Where to write logs/checkpoints/samples."},
    "validate.help": {"zh": "校验 config,但不启动训练。", "en": "Validate a config without running training."},
    "validate.ok": {"zh": "[green]通过[/] config 合法", "en": "[green]OK[/] config valid"},
    "info.help": {"zh": "展示 config 编译后的 argv 与显存估算(不训练)。", "en": "Show what a config would compile to, plus VRAM estimate (no training)."},
    "info.title": {"zh": "Config 概览", "en": "Config summary"},
    "info.compiled_argv": {"zh": "[bold]编译得到的命令行参数:[/bold]", "en": "[bold]Compiled argv:[/bold]"},
    "train.help": {"zh": "运行训练直至完成。按 Ctrl+C 优雅停止。", "en": "Run training to completion. Press Ctrl+C to stop gracefully."},
    "train.workspace_label": {"zh": "[dim]工作目录:[/dim] {ws}", "en": "[dim]workspace:[/dim] {ws}"},
    "train.process_label": {"zh": "[dim]进程:[/dim] {pid}  [dim]任务:[/dim] {job}", "en": "[dim]pid:[/dim] {pid}  [dim]job:[/dim] {job}"},
    "train.interrupt": {"zh": "\n[yellow]收到 Ctrl+C - 正在优雅停止训练...[/yellow]", "en": "\n[yellow]Ctrl+C - stopping training gracefully...[/yellow]"},
    "train.failed": {"zh": "[red]训练失败 (返回码 {rc})[/red]", "en": "[red]training failed (rc={rc})[/red]"},
    "train.ok": {"zh": "[green]通过[/] 训练完成", "en": "[green]OK[/] training complete"},

    # ── serve ──────────────────────────────────────────────────────
    "serve.help": {"zh": "运行 LoraHub HTTP API 服务(REST + WebSocket)。", "en": "Run the LoraHub HTTP API server (REST + WebSocket)."},
    "serve.api_extras_missing": {
        "zh": "[red]API 依赖未安装。[/red]请运行:pip install lorahub[api]",
        "en": "[red]API extras not installed.[/red] Run: pip install lorahub[api]",
    },
    "serve.banner": {
        "zh": "[bold]LoraHub API[/bold] http://{host}:{port}  (按 Ctrl+C 退出)",
        "en": "[bold]LoraHub API[/bold] http://{host}:{port}  (Ctrl+C to stop)",
    },
    "serve.host_help": {"zh": "监听地址。", "en": "Bind address."},
    "serve.port_help": {"zh": "监听端口。", "en": "Port to listen on."},
    "serve.reload_help": {"zh": "代码改动时自动重载(仅开发用)。", "en": "Auto-reload on code change (dev only)."},

    # ── init / scaffold ────────────────────────────────────────────
    "init.help": {"zh": "在当前目录生成入门 config。", "en": "Scaffold a starter config in the current directory."},
    "init.name_help": {"zh": "新 config 的名字(不带扩展名)。", "en": "Name for the new config (no extension)."},
    "init.template_help": {"zh": "复制哪个内置模板。--auto 时忽略。", "en": "Built-in template to copy. Ignored when --auto is used."},
    "init.auto_help": {"zh": "探测 GPU 与数据集,生成针对本机调优的 config。", "en": "Probe the GPU + dataset and write a config tuned to this machine."},
    "init.checkpoint_help": {"zh": "底模 .safetensors(--auto 必填)。", "en": "Base model .safetensors (required for --auto)."},
    "init.dataset_help": {"zh": "数据集目录(--auto 必填)。", "en": "Dataset directory (required for --auto)."},
    "init.vram_help": {"zh": "覆盖检测到的显存(MiB,如 8192),将跳过 nvidia-smi。", "en": "Override detected VRAM in MiB (e.g. 8192). Skips nvidia-smi."},
    "init.exists": {"zh": "[red]{path} 已存在[/red]", "en": "[red]{path} already exists[/red]"},
    "init.auto_requires": {
        "zh": "[red]--auto 需要同时传 --checkpoint 与 --dataset[/red]",
        "en": "[red]--auto requires --checkpoint and --dataset[/red]",
    },
    "init.created": {
        "zh": "[green]已创建[/green] {dst}\n[dim]架构[/dim] {arch}  [dim]rank[/dim] {rank}  [dim]批量[/dim] {batch}x{accum}  [dim]图片[/dim] {images}  [dim]重复[/dim] {repeats}",
        "en": "[green]created[/green] {dst}\n[dim]arch[/dim] {arch}  [dim]rank[/dim] {rank}  [dim]batch[/dim] {batch}x{accum}  [dim]images[/dim] {images}  [dim]repeats[/dim] {repeats}",
    },
    "init.unknown_template": {"zh": "[red]未知模板:{name}[/red]", "en": "[red]unknown template: {name}[/red]"},
    "init.copied": {"zh": "[green]已创建[/green] {dst}", "en": "[green]created[/green] {dst}"},

    # ── bootstrap-kohya ────────────────────────────────────────────
    "bootstrap.help": {"zh": "一键安装 kohya-ss/sd-scripts(克隆 + venv + PyTorch + 依赖 + xformers)。", "en": "One-shot install of kohya-ss/sd-scripts (clone + venv + PyTorch + deps + xformers)."},
    "bootstrap.target_help": {"zh": "克隆 sd-scripts 与 venv 的目录。", "en": "Where to clone sd-scripts and create its venv."},
    "bootstrap.cuda_help": {"zh": "CUDA wheel 后缀(cu118 / cu121 / cu124 / cu128)。", "en": "CUDA wheel suffix (cu118 / cu121 / cu124 / cu128)."},
    "bootstrap.torch_help": {"zh": "要安装的 PyTorch 版本。", "en": "PyTorch version to install."},
    "bootstrap.torchvision_help": {"zh": "要安装的 torchvision 版本。", "en": "torchvision version to install."},
    "bootstrap.no_xformers_help": {"zh": "跳过可选的 xformers 安装。", "en": "Skip the optional xformers install."},
    "bootstrap.force_help": {"zh": "重建受管环境；仅清理上次失败的克隆目录。", "en": "Rebuild the managed environment; clean only an interrupted clone."},
    "bootstrap.target_busy": {
        "zh": "[red]目标目录 {target} 非空。[/red] 加 --force 先清空,或用 --target 换路径。",
        "en": "[red]target {target} is not empty.[/red] Pass --force to wipe it first, or pick another path with --target.",
    },
    "bootstrap.banner": {
        "zh": "[bold]正在安装 kohya 到[/bold] {target}\n[dim]CUDA[/dim] {cuda}  [dim]torch[/dim] {torch}  [dim]xformers[/dim] {xformers}",
        "en": "[bold]Installing kohya into[/bold] {target}\n[dim]CUDA[/dim] {cuda}  [dim]torch[/dim] {torch}  [dim]xformers[/dim] {xformers}",
    },
    "bootstrap.step": {"zh": "[cyan]>[/cyan] {step}", "en": "[cyan]>[/cyan] {step}"},
    "bootstrap.failed": {
        "zh": "[red]bootstrap 失败,步骤:[/red] {step} [dim](返回码 {rc})[/dim]\n要从头重试,请运行 [bold]lorahub bootstrap-kohya --force[/bold]。",
        "en": "[red]bootstrap failed at step:[/red] {step} [dim](exit {rc})[/dim]\nRun [bold]lorahub bootstrap-kohya --force[/bold] to retry from scratch.",
    },
    "bootstrap.ok": {"zh": "[green]通过[/] kohya 已安装到 {target}", "en": "[green]OK[/] kohya installed at {target}"},
    "bootstrap.env_hint": {
        "zh": "[dim]请设置 LORAHUB_KOHYA_SD_SCRIPTS={target}(也可复制 .env.example 为 .env)。[/dim]",
        "en": "[dim]Set LORAHUB_KOHYA_SD_SCRIPTS={target} (or copy .env.example to .env).[/dim]",
    },

    # ── fetch-bangumi ──────────────────────────────────────────────
    "bangumi.help": {"zh": "从 BangumiBase HF 数据集下载单个角色图片。", "en": "Download a single character's images from a BangumiBase HF dataset."},
    "bangumi.repo_help": {"zh": "BangumiBase 仓库,如 'azurlaneanime' 或 'BangumiBase/azurlaneanime'。", "en": "BangumiBase repo, e.g. 'azurlaneanime' or 'BangumiBase/azurlaneanime'."},
    "bangumi.character_help": {"zh": "数字角色 id(如 '3'),省略则列出所有角色。", "en": "Numeric character id (e.g. '3'). Omit to list characters."},
    "bangumi.output_help": {"zh": "图片与 caption 的解压目录。", "en": "Where to unpack images and caption files."},
    "bangumi.limit_help": {"zh": "图片数量上限,适合做冒烟测试。", "en": "Cap on number of images. Useful for smoke testing."},
    "bangumi.preview_help": {"zh": "下载预览缩略图 1-8 而不是 dataset.zip。", "en": "Download preview thumbnails 1-8 instead of dataset.zip."},
    "bangumi.seed_help": {"zh": "为每张图生成空 .txt caption,默认开。", "en": "Seed empty .txt caption files next to each image. Default on."},
    "bangumi.list": {"zh": "[bold]{n} 个角色[/] 在 {repo}: {names}", "en": "[bold]{n} characters[/] in {repo}: {names}"},
    "bangumi.preview_line": {"zh": "[dim]预览 {i}[/dim]  {path}", "en": "[dim]preview {i}[/dim]  {path}"},
    "bangumi.fetched": {"zh": "[green]通过[/] 下载了 {n} 张图片 -> {dir}", "en": "[green]OK[/] {n} images -> {dir}"},
    "bangumi.license": {"zh": "[dim]许可证: {license}[/dim]", "en": "[dim]license: {license}[/dim]"},
    "bangumi.seed_warn": {
        "zh": "[yellow]已生成空白 .txt caption — 训练前请填写内容。[/yellow]",
        "en": "[yellow]Seeded empty .txt captions - fill them in before training.[/yellow]",
    },

    # ── tag (auto-tagger) ──────────────────────────────────────────
    "tag.help": {"zh": "自动给图片打标并写入 kohya 风格的 .txt caption。", "en": "Auto-tag images and write kohya-style .txt captions."},
    "tag.dir_help": {"zh": "要原地打标的图片目录。", "en": "Directory of images to tag in place."},
    "tag.tagger_help": {"zh": "选择 tagger:'wd14'(默认)或 'joytag'。", "en": "Which auto-tagger to use: 'wd14' (default) or 'joytag'."},
    "tag.model_help": {"zh": "WD tagger 在 Hugging Face 的模型 id(joytag 忽略)。", "en": "Hugging Face model id of the WD tagger (ignored for joytag)."},
    "tag.general_help": {"zh": "WD14 通用标分数阈值。", "en": "WD14 general-tag score threshold."},
    "tag.character_help": {"zh": "WD14 角色标分数阈值。", "en": "WD14 character-tag score threshold."},
    "tag.joytag_help": {"zh": "JoyTag 预测阈值(对全部 tag 单值)。", "en": "JoyTag predict threshold (single value across all tags)."},
    "tag.recursive_help": {"zh": "递归处理子目录。", "en": "Recurse into subdirectories."},
    "tag.overwrite_help": {"zh": "已有非空 caption 的图片也重新打标。", "en": "Re-tag images that already have a non-empty caption."},
    "tag.underscores_help": {"zh": "保留 tag 中的下划线而非替换为空格。", "en": "Keep underscores in tag names instead of spaces."},
    "tag.include_character_help": {"zh": "在 caption 中包含角色 tag(仅 WD14)。默认开。", "en": "Include character tags in the caption (WD14 only). Default on."},
    "tag.device_help": {"zh": "运行设备:'auto'(可用就用 CUDA)、'cuda'(强制 GPU)、'cpu'。", "en": "Runtime: 'auto' (CUDA if available), 'cuda' (force GPU), or 'cpu'."},
    "tag.not_a_dir": {"zh": "[red]不是目录:{path}[/red]", "en": "[red]not a directory: {path}[/red]"},
    "tag.unknown_tagger": {"zh": "[red]未知 tagger {name!r};可选 wd14 或 joytag[/red]", "en": "[red]unknown tagger {name!r}; expected wd14 or joytag[/red]"},
    "tag.loading_joytag": {"zh": "[dim]正在加载 fancyfeast/joytag(首次运行下载约 1.2GB)...[/dim]", "en": "[dim]loading fancyfeast/joytag (first run downloads ~1.2GB)...[/dim]"},
    "tag.loading_wd": {"zh": "[dim]正在加载 {model}(首次运行下载约 400MB)...[/dim]", "en": "[dim]loading {model} (first run downloads ~400MB)...[/dim]"},
    "tag.running_on": {"zh": "[dim]运行设备:{provider}[/dim]", "en": "[dim]running on {provider}[/dim]"},
    "tag.tagged_one": {"zh": "[dim]已打标[/dim] {name}", "en": "[dim]tagged[/dim] {name}"},
    "tag.ok": {"zh": "[green]通过[/] 共打标 {n} 张图片", "en": "[green]OK[/] tagged {n} images"},

    # ── anima-caption / caption ────────────────────────────────────
    "anima.help": {"zh": "把 *.txt caption 重写为 Anima 推荐格式(不调用 tagger)。", "en": "Rewrite *.txt captions to Anima's recommended layout (no tagger inference)."},
    "anima.rewrote_one": {"zh": "[dim]已重写[/dim] {name}", "en": "[dim]rewrote[/dim] {name}"},
    "anima.dry_run": {"zh": "[yellow]演练模式[/yellow](加 --overwrite 才会真正重写 caption)", "en": "[yellow]dry run[/yellow] (pass --overwrite to actually rewrite captions)"},
    "anima.ok": {"zh": "[green]通过[/] 共重写 {n} 个 caption", "en": "[green]OK[/] rewrote {n} caption(s)"},
    "caption.help": {"zh": "原地清理 booru 风格 caption(Illustrious / Pony / Animagine / NoobAI)。", "en": "Clean booru-style captions in place (Illustrious / Pony / Animagine / NoobAI)."},
    "caption.unknown_action": {"zh": "[red]未知 caption 子动作:{action}[/red]", "en": "[red]unknown caption action: {action}[/red]"},
    "caption.progress": {"zh": "[dim]{done}/{total}[/dim] {name}", "en": "[dim]{done}/{total}[/dim] {name}"},
    "caption.ok": {"zh": "[green]通过[/] 共重写 {n} 个 caption", "en": "[green]OK[/] rewrote {n} caption(s)"},

    # ── sweep (top-level grid expansion) ───────────────────────────
    "sweep.help": {"zh": "把网格 sweep 展开为多个 variant config 文件。", "en": "Expand a grid sweep into per-variant config files."},
    "sweep.need_axis": {"zh": "[red]至少需要一个 --axis 参数[/red]", "en": "[red]at least one --axis is required[/red]"},
    "sweep.bad_axis": {"zh": "[red]非法 --axis {spec!r};期望 'dotted.path=v1,v2,...'[/red]", "en": "[red]bad --axis spec {spec!r}; expected 'dotted.path=v1,v2,...'[/red]"},
    "sweep.empty_axis_path": {"zh": "[red]{spec!r} 中 axis 路径为空[/red]", "en": "[red]empty axis path in {spec!r}[/red]"},
    "sweep.no_values": {"zh": "[red]axis {path!r} 没有任何取值[/red]", "en": "[red]axis {path!r} has no values[/red]"},
    "sweep.dry_run_header": {"zh": "[bold]sweep[/bold] {n} 个 variant [dim](演练模式)[/dim]", "en": "[bold]sweep[/bold] {n} variant(s) [dim](dry run)[/dim]"},
    "sweep.variant_invalid": {"zh": "[red]variant {name!r} 校验失败:{err}[/red]", "en": "[red]variant {name!r} fails schema validation: {err}[/red]"},
    "sweep.ok": {"zh": "[green]通过[/] 写入 {n} 个 variant 到 {dir}", "en": "[green]OK[/] wrote {n} variant(s) to {dir}"},
    "sweep.manifest": {"zh": "[dim]清单文件:[/dim] {path}", "en": "[dim]manifest:[/dim] {path}"},

    # ── jobs sub-app ───────────────────────────────────────────────
    "jobs.help": {"zh": "无需打开 web UI 也能查看与管理训练任务。", "en": "Inspect and manage training jobs without opening the web UI."},
    "jobs.ls.empty": {"zh": "[dim]暂无任务[/dim]", "en": "[dim]no jobs[/dim]"},
    "jobs.ls.col_id": {"zh": "id", "en": "id"},
    "jobs.ls.col_state": {"zh": "状态", "en": "state"},
    "jobs.ls.col_name": {"zh": "名称", "en": "name"},
    "jobs.ls.col_workspace": {"zh": "工作目录", "en": "workspace"},
    "jobs.ls.col_created": {"zh": "创建时间", "en": "created"},
    "jobs.unknown_state": {"zh": "[red]未知状态 {state!r}[/red]", "en": "[red]unknown state {state!r}[/red]"},
    "jobs.cancel.not_queued": {
        "zh": "[yellow]任务 {id} 当前状态为 {state},不是 queued — 用 `lorahub jobs kill` 终止运行中的任务。[/yellow]",
        "en": "[yellow]job {id} is {state}, not queued — use `lorahub jobs kill` to terminate a running job.[/yellow]",
    },
    "jobs.cancel.ok": {"zh": "[green]已取消[/green] {id}", "en": "[green]canceled[/green] {id}"},
    "jobs.kill.no_pid": {"zh": "[yellow]任务 {id} 无 PID 记录(状态 {state})[/yellow]", "en": "[yellow]job {id} has no recorded pid (state={state})[/yellow]"},
    "jobs.kill.marked": {"zh": "[green]已标记为 canceled[/green]", "en": "[green]marked canceled[/green]"},
    "jobs.kill.sent": {"zh": "[dim]已向 pid {pid} 发送 {sig}[/dim]", "en": "[dim]sent {sig} to pid {pid}[/dim]"},
    "jobs.kill.gone": {"zh": "[yellow]pid {pid} 已不存在[/yellow]", "en": "[yellow]pid {pid} no longer alive[/yellow]"},
    "jobs.kill.no_perm": {"zh": "[red]无法对 pid {pid} 发信号:{err}[/red]", "en": "[red]cannot signal pid {pid}: {err}[/red]"},
    "jobs.kill.escalating": {"zh": "[yellow]worker 未退出;升级为 SIGKILL[/yellow]", "en": "[yellow]worker didn't exit; escalating to SIGKILL[/yellow]"},
    "jobs.kill.ok": {"zh": "[green]已终止[/green] {id}", "en": "[green]killed[/green] {id}"},
    "jobs.resume.ok": {"zh": "[green]已恢复[/green] → 新任务 {id}", "en": "[green]resumed[/green] → new job {id}"},
    "jobs.rerun.ok": {"zh": "[green]已重跑[/green] → 新任务 {id}", "en": "[green]rerun[/green] → new job {id}"},
    "jobs.no_match": {"zh": "[red]没有任务匹配 {id!r}[/red]", "en": "[red]no job matches {id!r}[/red]"},
    "jobs.ambiguous": {"zh": "[red]后缀 {id!r} 模糊,匹配 {n} 个任务:[/red]", "en": "[red]ambiguous suffix {id!r} matches {n} jobs:[/red]"},
    "jobs.http_error": {"zh": "[red]HTTP {code}[/red] {url}: {body}", "en": "[red]HTTP {code}[/red] {url}: {body}"},
    "jobs.unreachable": {
        "zh": "[red]无法连接[/red] {url}: {reason}\n[yellow]`lorahub serve` 是否在本机上运行?[/yellow]",
        "en": "[red]could not reach[/red] {url}: {reason}\n[yellow]is `lorahub serve` running on this host?[/yellow]",
    },

    # ── service sub-app ────────────────────────────────────────────
    "service.help": {"zh": "管理 LoraHub API 守护进程(start / stop / status / logs / enable)。", "en": "Manage the LoraHub API daemon (start/stop/status/logs/enable)."},
    "service.start.host_help": {"zh": "监听地址。", "en": "Bind address."},
    "service.start.port_help": {"zh": "监听端口。0 表示自动选空闲端口(默认)。", "en": "Port to listen on. 0 picks a free port (default)."},
    "service.start.foreground_help": {"zh": "前台运行 uvicorn,而不是后台脱离。", "en": "Run uvicorn in this terminal instead of detached."},
    "service.already_running": {"zh": "[yellow]守护进程已在运行[/] pid={pid}{port}", "en": "[yellow]already running[/] pid={pid}{port}"},
    "service.foreground_banner": {"zh": "[bold]LoraHub[/bold] 前台运行 http://{host}:{port}", "en": "[bold]LoraHub[/bold] foreground http://{host}:{port}"},
    "service.started": {"zh": "已启动 pid={pid} port={port}", "en": "started pid={pid} port={port}"},
    "service.log_path": {"zh": "日志: {path}", "en": "log: {path}"},
    "service.healthy": {"zh": "[green]健康[/] http://{host}:{port}", "en": "[green]healthy[/] http://{host}:{port}"},
    "service.remote_auth": {
        "zh": "远程访问认证已启用。令牌文件: {path}",
        "en": "remote access authentication enabled. Token file: {path}",
    },
    "service.remote_auth_env": {
        "zh": "远程访问认证已启用。令牌来自 LORAHUB_API_TOKEN。",
        "en": "remote access authentication enabled by LORAHUB_API_TOKEN.",
    },
    "service.health_timeout": {"zh": "[yellow]守护进程已启动,但 30 秒内 /api/health 未响应。请查看 {log}[/]", "en": "[yellow]daemon launched but /api/health did not answer within 30s. Check {log}[/]"},
    "service.stop.timeout_help": {"zh": "优雅关闭超时(秒),超过则发送 SIGKILL。", "en": "Seconds to wait for graceful shutdown before SIGKILL."},
    "service.stop.not_running": {"zh": "未在运行", "en": "not running"},
    "service.stop.failed": {"zh": "[red]无法停止 pid {pid}:[/] {err}", "en": "[red]failed to stop pid {pid}:[/] {err}"},
    "service.stopped": {"zh": "已停止 pid={pid}", "en": "stopped pid={pid}"},
    "service.status.stopped": {"zh": "[dim]已停止[/]", "en": "[dim]stopped[/]"},
    "service.status.running": {"zh": "[green]运行中[/] pid={pid} {port_label} {health_label}", "en": "[green]running[/] pid={pid} {port_label} {health_label}"},
    "service.status.healthy": {"zh": "[green]健康[/]", "en": "[green]healthy[/]"},
    "service.status.unhealthy": {"zh": "[yellow]启动中/不健康[/]", "en": "[yellow]starting/unhealthy[/]"},
    "service.logs.empty": {"zh": "[dim]还没有日志:[/dim] {path}", "en": "[dim]no log yet:[/dim] {path}"},
    "service.enable.windows": {
        "zh": "[red]Windows 不支持 `service enable`。[/]\n请使用任务计划程序。示例命令:\n  schtasks /Create /SC ONLOGON /TN LoraHub /TR \"{exe} -m lorahub service start --foreground --host {host} --port {port}\"",
        "en": "[red]Windows isn't supported by `service enable`.[/]\nUse Task Scheduler. Sample invocation:\n  schtasks /Create /SC ONLOGON /TN LoraHub /TR \"{exe} -m lorahub service start --foreground --host {host} --port {port}\"",
    },
    "service.enable.perm": {
        "zh": "[red]写入 {path} 权限不足。[/]\n请加 sudo 重试: sudo lorahub service enable",
        "en": "[red]permission denied writing {path}.[/]\nRe-run with sudo: sudo lorahub service enable",
    },
    "service.enable.ok": {"zh": "[green]已启用[/] {path}", "en": "[green]enabled[/] {path}"},
    "service.enable.systemd_hint": {"zh": "状态: systemctl status lorahub\n日志: journalctl -u lorahub -f", "en": "status: systemctl status lorahub\nlogs:   journalctl -u lorahub -f"},
    "service.disable.windows": {"zh": "[yellow]Windows: 用如下命令删除任务[/] schtasks /Delete /TN LoraHub /F", "en": "[yellow]Windows: remove the task with[/] schtasks /Delete /TN LoraHub /F"},
    "service.disable.sudo": {"zh": "[red]需要 sudo 才能删除 {path}[/]", "en": "[red]sudo required to remove {path}[/]"},
    "service.disable.ok": {"zh": "[green]已停用[/]", "en": "[green]disabled[/]"},

    # ── system sub-app ─────────────────────────────────────────────
    "system.help": {"zh": "查看本机 CPU / GPU / 内存状态。", "en": "Inspect local CPU / GPU / memory state."},
    "system.gpu.no_gpus": {"zh": "[yellow]未检测到 GPU[/yellow]", "en": "[yellow]no GPUs detected[/yellow]"},
    "system.gpu.no_smi": {"zh": "[dim]PATH 上没有 nvidia-smi;只能采集 CPU 状态。[/dim]", "en": "[dim]nvidia-smi not on PATH; only CPU stats are available.[/dim]"},
    "system.col.idx": {"zh": "序号", "en": "idx"},
    "system.col.name": {"zh": "型号", "en": "name"},
    "system.col.mem": {"zh": "显存", "en": "mem"},
    "system.col.util": {"zh": "利用率", "en": "util"},
    "system.col.temp": {"zh": "温度", "en": "temp"},
    "system.col.driver": {"zh": "驱动", "en": "driver"},
    "system.info.host": {"zh": "[bold]主机:[/bold] {hostname}  ({system} {release})", "en": "[bold]host:[/bold] {hostname}  ({system} {release})"},
    "system.info.python": {"zh": "  Python: {version}", "en": "  python: {version}"},
    "system.info.cpu": {"zh": "  CPU: {logical} 逻辑核 / {physical} 物理核 @ 负载 {usage:.1f}%", "en": "  CPU: {logical} logical / {physical} physical @ {usage:.1f}% load"},
    "system.info.ram": {"zh": "  内存: {used:.1f} / {total:.1f} GiB ({percent:.1f}%)", "en": "  RAM: {used:.1f} / {total:.1f} GiB ({percent:.1f}%)"},
    "system.info.gpus": {"zh": "  GPU: {n} 张 ({names})", "en": "  GPUs: {n} ({names})"},
    "system.info.disks": {"zh": "  磁盘: {n} 个挂载点", "en": "  Disks: {n} mount points"},

    # ── sweep sub-app (HTTP) ───────────────────────────────────────
    "sweepapp.help": {"zh": "向调度器提交超参 sweep。", "en": "Submit hyperparameter sweeps to the scheduler."},
    "sweepapp.not_a_file": {"zh": "[red]不是文件:[/red] {path}", "en": "[red]not a file:[/red] {path}"},
    "sweepapp.yaml_error": {"zh": "[red]YAML 解析错误:[/red] {err}", "en": "[red]yaml parse error:[/red] {err}"},
    "sweepapp.yaml_not_mapping": {"zh": "[red]顶层 YAML 必须是映射[/red]", "en": "[red]top-level YAML value must be a mapping[/red]"},
    "sweepapp.unreachable": {
        "zh": "[red]无法连接[/red] {url}: {reason}\n[yellow]`lorahub serve` 是否在运行?[/yellow]",
        "en": "[red]could not reach[/red] {url}: {reason}\n[yellow]is `lorahub serve` running?[/yellow]",
    },
    "sweepapp.empty": {"zh": "[dim]暂无 sweep[/dim]", "en": "[dim]no sweeps[/dim]"},

    # ── service / jobs / system / sweep sub-command short helps ────
    "service.start.help": {"zh": "启动 API 守护进程。", "en": "Start the API daemon."},
    "service.stop.help": {"zh": "停止 API 守护进程。", "en": "Stop the API daemon."},
    "service.restart.help": {"zh": "重启守护进程(如已在运行先停止)。", "en": "Stop the daemon (if running) and start a fresh one."},
    "service.status.help": {"zh": "查看守护进程是否在运行及监听端口。", "en": "Show whether the daemon is running and on which port."},
    "service.logs.help": {"zh": "打印守护进程日志。", "en": "Print the daemon's log file."},
    "service.enable.help": {"zh": "把 LoraHub 注册为系统服务(仅 Linux/macOS)。", "en": "Register LoraHub as a system service (Linux/macOS only)."},
    "service.disable.help": {"zh": "卸载已注册的系统服务。", "en": "Remove the registered system service."},
    "service.install_unit.help": {"zh": "打印 systemd unit / launchd plist / 计划任务样例。", "en": "Print the systemd unit / launchd plist / Task Scheduler stub."},
    "jobs.ls.help": {"zh": "按创建时间倒序列出任务。", "en": "List jobs sorted by creation time, newest first."},
    "jobs.show.help": {"zh": "打印一个任务的完整记录(状态、指标、错误等)。", "en": "Print one job's full record (state, metrics, metadata, error)."},
    "jobs.cancel.help": {"zh": "取消排队中的任务。运行中的请用 `kill`。", "en": "Cancel a queued job. Use ``kill`` for running jobs."},
    "jobs.kill.help": {"zh": "按 PID 停止运行中的任务,然后在 store 中标记为 canceled。", "en": "Stop a running job by PID, then mark it canceled in the store."},
    "jobs.resume.help": {"zh": "从最近 checkpoint 恢复一个被中断的任务。", "en": "Resume an interrupted job from its last checkpoint."},
    "jobs.rerun.help": {"zh": "用同一份 config 重新启动一个已结束的任务。", "en": "Re-launch a finished job with the same config."},
    "system.gpu.help": {"zh": "打印 GPU 一次性快照:型号、显存、利用率、温度。", "en": "Print one-shot GPU info: name, memory, utilisation, temp, processes."},
    "system.info.help": {"zh": "打印完整主机快照(CPU + 内存 + 磁盘 + 网络)。", "en": "Print the full host snapshot (CPU + memory + disks + network)."},
    "system.errors.help": {"zh": "查看本地错误上报记录(最近 N 条)。", "en": "Show recent entries from the local error-report registry."},
    "system.errors.tail_help": {"zh": "显示最近多少条记录。", "en": "How many recent rows to print."},
    "system.errors.severity_help": {"zh": "按严重程度过滤:fatal / error / warn / info。", "en": "Filter by severity: fatal / error / warn / info."},
    "system.errors.source_help": {"zh": "按来源过滤,如 backend.job、frontend.render。", "en": "Filter by source, e.g. backend.job, frontend.render."},
    "system.errors.empty": {"zh": "[dim]暂无错误记录[/dim]", "en": "[dim]no error reports[/dim]"},
    "system.errors.col_time": {"zh": "时间", "en": "time"},
    "system.errors.col_severity": {"zh": "严重", "en": "severity"},
    "system.errors.col_source": {"zh": "来源", "en": "source"},
    "system.errors.col_title": {"zh": "标题", "en": "title"},
    "system.errors.col_id": {"zh": "ID 后缀", "en": "id suffix"},
    "system.errors_show.help": {"zh": "查看一条错误的完整堆栈与上下文。", "en": "Print one error report's full stack + context as JSON."},
    "system.errors_show.id_help": {"zh": "错误 ID(完整或末尾 12 位)。", "en": "Error id (full string or 12-char suffix)."},
    "system.errors_show.no_match": {"zh": "[red]未找到匹配 {id!r} 的错误记录[/red]", "en": "[red]no error report matches {id!r}[/red]"},
    "system.errors_show.ambiguous": {"zh": "[red]后缀 {id!r} 匹配 {n} 条记录:[/red]", "en": "[red]suffix {id!r} matches {n} reports:[/red]"},
    "system.errors_export.help": {"zh": "导出错误记录到 ndjson 文件。", "en": "Export the error registry as newline-delimited JSON."},
    "system.errors_export.output_help": {"zh": "输出 ndjson 文件路径。", "en": "Where to write the ndjson file."},
    "system.errors_export.limit_help": {"zh": "导出最多多少条。", "en": "Cap on rows to export."},
    "system.errors_export.ok": {"zh": "[green]通过[/] 导出 {n} 条记录到 {path}", "en": "[green]OK[/] wrote {n} report(s) to {path}"},
    "system.errors_clear.help": {"zh": "清空本地错误上报记录(不可恢复)。", "en": "Drop every row in the error registry (not recoverable)."},
    "system.errors_clear.yes_help": {"zh": "确认清空(必填)。", "en": "Confirm the destructive action (required)."},
    "system.errors_clear.confirm_required": {
        "zh": "[red]需要 --yes 确认才能清空错误记录。[/red]",
        "en": "[red]refused to clear without --yes confirmation.[/red]",
    },
    "system.errors_clear.ok": {"zh": "[green]通过[/] 清空了 {n} 条记录", "en": "[green]OK[/] cleared {n} report(s)"},
    "sweepapp.submit.help": {"zh": "提交一个 sweep YAML 到 ``POST /api/sweeps``。", "en": "Submit a sweep YAML to ``POST /api/sweeps`` and print the response."},
    "sweepapp.ls.help": {"zh": "列出运行中服务上的所有 sweep。", "en": "List every sweep on the running server."},

    # ── ref-extract (差异训练参考图自动生成 - 仅 canny) ─────────────
    "ref_extract.help": {
        "zh": "为差异训练自动生成 Canny 边缘参考图。每张目标图过 cv2 Canny,输出按主名同名写到 dst,直接接 LoraHub 数据集子集的「参考图目录」。\n\n[dim]更复杂的参考图(DWPose 骨架 / 线稿 / 深度图)请直接走 ComfyUI 生态(controlnet_aux 节点),完成后把目录路径填到「参考图目录」即可。本程序不集成那些重型预处理器。[/dim]",
        "en": "Auto-generate Canny edge reference images for conditioning training. Each target is run through cv2 Canny; outputs land in dst with same stem so the LoraHub dataset subset's conditioning_data_dir picks them up.\n\n[dim]Heavier processors (DWPose skeleton / lineart / depth) are NOT integrated here — generate those via ComfyUI (controlnet_aux nodes) and just point conditioning_data_dir at the result.[/dim]",
    },
    "ref_extract.processor.help": {
        "zh": "预处理器:目前仅支持 canny。其它 (DWPose / 线稿 / 深度) 请用 ComfyUI 生成。",
        "en": "Processor: only canny is supported. Use ComfyUI for DWPose / lineart / depth.",
    },
    "ref_extract.src.help": {
        "zh": "源(target)图目录。", "en": "Source (target) image directory.",
    },
    "ref_extract.dst.help": {
        "zh": "参考图输出目录。已存在的同名文件默认跳过(除非 --overwrite)。",
        "en": "Output directory for reference images. Existing same-stem files are skipped unless --overwrite.",
    },
    "ref_extract.overwrite.help": {
        "zh": "覆盖目标目录里已存在的同名 ref 文件。", "en": "Overwrite same-stem reference files already in dst.",
    },
    "ref_extract.recursive.help": {
        "zh": "递归处理 src 子目录,镜像层级到 dst。", "en": "Recurse into subdirs of src, mirroring layout in dst.",
    },
    "ref_extract.dep_missing": {
        "zh": "[red]缺少依赖:[/red] {pkg}\n请在当前 venv 安装:\n  [yellow]pip install {pkg}[/yellow]",
        "en": "[red]missing dependency:[/red] {pkg}\nInstall in this venv:\n  [yellow]pip install {pkg}[/yellow]",
    },
    "ref_extract.dep_missing_real": {
        "zh": "[red]缺少子依赖:[/red] {missing}\n请安装:\n  [yellow]pip install {missing}[/yellow]\n[dim]原始错误: {err}[/dim]",
        "en": "[red]missing sub-dependency:[/red] {missing}\nInstall:\n  [yellow]pip install {missing}[/yellow]\n[dim]original error: {err}[/dim]",
    },
    "ref_extract.canny_low.help": {
        "zh": "Canny 低阈值。", "en": "Canny low threshold.",
    },
    "ref_extract.canny_high.help": {
        "zh": "Canny 高阈值。", "en": "Canny high threshold.",
    },
    "ref_extract.start": {
        "zh": "[bold]ref-extract[/bold] processor=[cyan]{processor}[/cyan] src={src} → dst={dst}",
        "en": "[bold]ref-extract[/bold] processor=[cyan]{processor}[/cyan] src={src} → dst={dst}",
    },
    "ref_extract.scanned": {
        "zh": "扫描到 [bold]{n}[/bold] 张图待处理(已跳过 {skipped} 张已存在的)。",
        "en": "Found [bold]{n}[/bold] images to process ({skipped} skipped as already present).",
    },
    "ref_extract.processing": {"zh": "处理中…", "en": "processing..."},
    "ref_extract.failed": {"zh": "[red]失败[/red] {path}: {err}", "en": "[red]failed[/red] {path}: {err}"},
    "ref_extract.done": {
        "zh": "[green]完成[/green] 成功 {ok} / 失败 {fail} / 跳过 {skipped}",
        "en": "[green]done[/green] ok={ok} fail={fail} skipped={skipped}",
    },
}


def t(key: str, /, **fmt_args: object) -> str:
    """Look up a CLI string in the active language.

    Falls back to English when the key is missing in the current
    locale, then to the key itself when neither locale knows it —
    a runtime KeyError here would mask the real error the user is
    trying to read.
    """
    bundle = MESSAGES.get(key)
    if bundle is None:
        return key
    raw = bundle.get(_current_lang) or bundle.get("en") or key
    if fmt_args:
        try:
            return raw.format(**fmt_args)
        except (KeyError, IndexError):
            return raw
    return raw


__all__ = ["Lang", "current_lang", "set_lang", "t"]
