"""Rich-based terminal interface for the top-level ``lorahub`` command."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from lorahub import __version__


@dataclass(frozen=True)
class TuiAction:
    key: str
    title: str
    command: tuple[str, ...]
    description: str
    safe_to_run: bool = True


@dataclass(frozen=True)
class TuiGroup:
    title: str
    accent: str
    actions: tuple[TuiAction, ...]


GROUPS: tuple[TuiGroup, ...] = (
    TuiGroup(
        "服务",
        "cyan",
        (
            TuiAction("1", "启动服务", ("service", "start"), "后台启动 API 与前端服务。"),
            TuiAction("2", "服务状态", ("service", "status"), "查看端口、PID 与健康状态。"),
            TuiAction("3", "重启服务", ("service", "restart"), "复用上次端口重启。"),
            TuiAction("4", "查看日志", ("service", "logs"), "输出最近服务日志。"),
        ),
    ),
    TuiGroup(
        "维护",
        "green",
        (
            TuiAction("5", "环境自检", ("doctor",), "检查 Python、Node、构建产物与后端。"),
            TuiAction("6", "系统信息", ("system", "info"), "查看 CPU、内存、磁盘与 GPU 概览。"),
            TuiAction("7", "GPU 快照", ("system", "gpu"), "查看 GPU 显存、利用率、温度。"),
            TuiAction("8", "更新 Dev", ("manage", "update"), "拉取 dev、安装依赖并重建。", False),
            TuiAction("9", "升级正式版", ("manage", "upgrade"), "切换到最新发布 tag。", False),
        ),
    ),
    TuiGroup(
        "训练",
        "magenta",
        (
            TuiAction("10", "任务列表", ("jobs", "ls"), "查看最近训练任务。"),
            TuiAction("11", "校验配置", ("validate", "<config.yaml>"), "验证配置但不启动训练。", False),
            TuiAction("12", "配置概览", ("info", "<config.yaml>"), "查看编译参数与显存估算。", False),
            TuiAction("13", "启动训练", ("train", "<config.yaml>"), "用配置启动一次训练。", False),
        ),
    ),
    TuiGroup(
        "数据",
        "yellow",
        (
            TuiAction("14", "批量打标", ("tag", "<dataset_dir>"), "给图片目录生成标签文本。", False),
            TuiAction("15", "整理 Caption", ("caption", "normalize", "<dataset_dir>"), "清理、映射与规整 caption。", False),
            TuiAction("16", "Anima Caption", ("anima-caption", "<caption_dir>"), "转换为 Anima caption 结构。", False),
            TuiAction("17", "参考图提取", ("ref-extract", "<src>", "<dst>"), "生成差异训练参考图。", False),
        ),
    ),
)


def all_actions() -> list[TuiAction]:
    return [action for group in GROUPS for action in group.actions]


def command_text(command: Iterable[str]) -> str:
    return "lorahub " + " ".join(command)


def run_tui(console: Console | None = None) -> None:
    """Open the operator TUI.

    This intentionally stays on Rich instead of Textual: zero new dependency,
    instant startup, and safe behaviour inside SSH sessions.
    """
    console = console or Console()
    actions = {action.key: action for action in all_actions()}

    while True:
        console.clear()
        _render_home(console)
        choice = Prompt.ask(
            "[bold]选择[/] 编号执行,输入 [cyan]h[/cyan] 查看帮助,[cyan]q[/cyan] 退出",
            default="q",
            console=console,
        ).strip().lower()

        if choice in {"q", "quit", "exit"}:
            return
        if choice in {"h", "help", "?"}:
            _show_help(console)
            continue
        action = actions.get(choice)
        if action is None:
            _pause(console, "[yellow]未知选择[/yellow]")
            continue
        _handle_action(console, action)


def _render_home(console: Console) -> None:
    title = Text("LoRaHub", style="bold cyan")
    subtitle = Text(f"v{__version__}  训练、服务、维护与数据工具", style="dim")
    header = Panel(
        Align.center(Group(title, subtitle), vertical="middle"),
        box=box.ROUNDED,
        padding=(1, 2),
        border_style="cyan",
    )
    console.print(header)
    console.print()
    console.print(Columns((_group_panel(group) for group in GROUPS), equal=True, expand=True))
    console.print()
    console.print(
        Panel(
            "[dim]提示:[/] 带占位符的命令会先显示可复制命令,不会直接执行。"
            " 原有命令仍可按脚本方式使用,例如 [cyan]lorahub service start --port 18765[/cyan]。",
            box=box.SIMPLE,
            border_style="dim",
        )
    )


def _group_panel(group: TuiGroup) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True, style=f"bold {group.accent}")
    table.add_column(ratio=1)
    for action in group.actions:
        table.add_row(action.key, f"[bold]{action.title}[/bold]\n[dim]{action.description}[/dim]")
    return Panel(
        table,
        title=f"[{group.accent}]{group.title}[/]",
        box=box.ROUNDED,
        border_style=group.accent,
        padding=(1, 1),
    )


def _handle_action(console: Console, action: TuiAction) -> None:
    cmd = command_text(action.command)
    console.clear()
    console.print(
        Panel(
            f"[bold]{action.title}[/bold]\n\n[cyan]{cmd}[/cyan]\n\n{action.description}",
            title="命令",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )
    if not action.safe_to_run:
        _pause(console, "[dim]该命令需要参数或会修改环境,请复制后按需执行。[/dim]")
        return

    answer = Prompt.ask("执行该命令?", choices=["y", "n"], default="y", console=console)
    if answer != "y":
        return
    console.print()
    rc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "lorahub", *action.command],
        check=False,
    ).returncode
    _pause(console, f"[dim]命令结束,退出码 {rc}[/dim]")


def _show_help(console: Console) -> None:
    console.clear()
    console.print(
        Panel(
            "LoRaHub TUI 是 CLI 的交互式入口。\n\n"
            "无参数运行 [cyan]lorahub[/cyan] 会在交互式终端进入此界面。\n"
            "脚本、CI、管道环境仍输出普通帮助。\n"
            "可用 [cyan]lorahub --no-tui[/cyan] 强制禁用,或 [cyan]lorahub --tui[/cyan] 强制打开。",
            title="帮助",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )
    _pause(console)


def _pause(console: Console, message: str = "[dim]按 Enter 返回[/dim]") -> None:
    console.print(message)
    Prompt.ask("", default="", show_default=False, console=console)
