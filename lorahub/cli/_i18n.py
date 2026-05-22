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
    "manage.path.shim": {"zh": "用户 PATH 软链:", "en": "user-PATH shim:"},
    "manage.path.not_on_path": {"zh": "不在 PATH 上", "en": "not on PATH"},
    "manage.path.none": {"zh": "未安装", "en": "none"},
    "manage.path.exists": {"zh": "已存在", "en": "exists"},
    "manage.path.absent": {"zh": "缺失", "en": "absent"},

    "manage.install.help": {
        "zh": "把 `lorahub` 软链 / 启动器写入用户 PATH。",
        "en": "Add a ``lorahub`` shim to the user PATH.",
    },
    "manage.install.no_venv_entry": {
        "zh": "[red]当前 venv 中没有 lorahub 入口[/]\n请先运行 scripts/install.{sh,bat} 让 .venv/bin/lorahub 存在。",
        "en": "[red]no lorahub entry in the active venv[/]\nRun scripts/install.{sh,bat} first so .venv/bin/lorahub exists.",
    },
    "manage.install.setx_failed": {
        "zh": "[yellow]软链写入 {shim},但 setx PATH 失败:[/] {err}\n请手动把 {shim_dir} 加到用户 PATH。",
        "en": "[yellow]wrote shim {shim}, but setx PATH failed:[/] {err}\nAdd this to your user PATH manually: {shim_dir}",
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
        "zh": "从用户 PATH 中移除 `lorahub` 软链。",
        "en": "Remove the ``lorahub`` shim from the user PATH.",
    },
    "manage.uninstall.no_shim": {
        "zh": "[dim]{shim} 处没有软链[/]",
        "en": "[dim]no shim at {shim}[/]",
    },
    "manage.uninstall.removed": {
        "zh": "[green]已移除[/] {shim}",
        "en": "[green]removed[/] {shim}",
    },
    "manage.uninstall.dir_hint": {
        "zh": "[dim]提示:软链所在目录仍在你的用户 PATH 上。[/]\n如想彻底清理,请通过 设置 → 环境变量 移除。",
        "en": "[dim]note: the shim directory is still on your user PATH.[/]\nRemove it via Settings → Environment Variables if you want a clean slate.",
    },

    "manage.update.help": {
        "zh": "拉取 origin/main 最新代码、重装 Python 依赖、重建前端。",
        "en": "Pull the latest commits from origin/main, reinstall deps, rebuild SPA.",
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
    "manage.build.complete": {
        "zh": "[green]构建完成[/]",
        "en": "[green]build complete[/]",
    },

    # Phase prefixes for the streaming progress emit (git / deps / build / done).
    "manage.phase.git": {"zh": "[blue]git[/]", "en": "[blue]git[/]"},
    "manage.phase.deps": {"zh": "[cyan]依赖[/]", "en": "[cyan]deps[/]"},
    "manage.phase.build": {"zh": "[magenta]构建[/]", "en": "[magenta]build[/]"},
    "manage.phase.done": {"zh": "[green]完成[/]", "en": "[green]done[/]"},
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
