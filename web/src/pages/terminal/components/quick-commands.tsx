import { Play, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"

interface QuickCommandsProps {
  disabled: boolean
  /** Splice the command into the input box without running. */
  onPick: (command: string) => void
  /** Drop straight into the run path, skipping the input box. */
  onRun: (command: string) => void
}

const PRESETS: Array<{ label: string; command: string; hint?: string }> = [
  { label: "pip list", command: "pip list", hint: "已安装包" },
  { label: "pip freeze", command: "pip freeze", hint: "可粘贴的固定版本" },
  {
    label: "pip outdated",
    command: "pip list --outdated",
    hint: "可升级的包",
  },
  {
    label: "依赖检查",
    command: "pip check",
    hint: "检查已安装包的依赖一致性",
  },
  {
    label: "uv pip list",
    command: "uv pip list",
    hint: "通过 uv 直接列包（适合 uv venv）",
  },
  {
    label: "ensurepip",
    command: "python -m ensurepip --upgrade",
    hint: "在 venv 内补装 pip",
  },
  { label: "python -V", command: "python -V", hint: "解释器版本" },
  {
    label: "GPU 状态",
    command: "lorahub system gpu",
    hint: "GPU、驱动与显存状态",
  },
]

export function QuickCommands({ disabled, onPick, onRun }: QuickCommandsProps) {
  return (
    <div className="rounded-[5px] border border-border/60 bg-muted/15 px-3 py-2.5 space-y-1.5">
      <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
        快捷命令
      </div>
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((preset) => (
          <div
            key={preset.command}
            className="inline-flex items-center rounded-[3px] border border-border/60 bg-background overflow-hidden"
          >
            <button
              type="button"
              onClick={() => onPick(preset.command)}
              disabled={disabled}
              className="px-2 py-1 text-[11px] hover:bg-muted/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title={preset.hint ? `填入：${preset.command}` : `填入：${preset.command}`}
            >
              <span className="inline-flex items-center gap-1">
                <Plus className="size-2.5" />
                {preset.label}
              </span>
            </button>
            <span className="h-full w-px bg-border/60" aria-hidden />
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => onRun(preset.command)}
              disabled={disabled}
              title={`立即运行：${preset.command}`}
              className="rounded-none border-0 h-7 w-6"
            >
              <Play className="size-2.5" />
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
