import { ChevronDown, ChevronUp, Eye, Radio } from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ReferenceRun } from "./reference-run"
import type { PanelState, ViewMode } from "./view-mode"
import type { XMode } from "./x-axis-mode"

const MODE_BUTTONS: {
  key: ViewMode
  label: string
  description: string
  icon: typeof Eye
}[] = [
  {
    key: "live",
    label: "实时",
    description: "聚焦当下：是否在收敛 · 是否过拟合 · 是否按时跑完",
    icon: Radio,
  },
  {
    key: "postmortem",
    label: "复盘",
    description: "事后分析：所有面板默认展开",
    icon: Eye,
  },
]

export function ViewModeSwitcher({
  mode,
  panels,
  isTerminal,
  xMode,
  epochAvailable,
  referenceRun,
  isCurrentReference,
  onSelectMode,
  onTogglePanel,
  onSelectXMode,
  onPinReference,
  onClearReference,
}: {
  mode: ViewMode
  panels: PanelState
  isTerminal: boolean
  xMode: XMode
  epochAvailable: boolean
  referenceRun: ReferenceRun | null
  isCurrentReference: boolean
  onSelectMode: (mode: ViewMode) => void
  onTogglePanel: (key: keyof PanelState) => void
  onSelectXMode: (mode: XMode) => void
  onPinReference: () => void
  onClearReference: () => void
}) {
  return (
    <div className="rounded-[6px] border border-border/60 bg-card/50 px-3.5 py-2 flex items-center flex-wrap gap-3">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80 mr-1">
          视图模式
        </span>
        {MODE_BUTTONS.map((m) => {
          const Icon = m.icon
          const active = mode === m.key
          return (
            <button
              key={m.key}
              type="button"
              onClick={() => onSelectMode(m.key)}
              title={m.description}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-[3px] border px-2 py-0.5 text-[11px] transition-colors",
                active
                  ? "border-primary/45 bg-primary/15 text-foreground"
                  : "border-border/55 bg-background/60 text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="size-3" />
              {m.label}
              {active && m.key === "live" && !isTerminal && (
                <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </button>
          )
        })}
        {mode === "custom" && (
          <span className="inline-flex items-center rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.16em] text-amber-700 dark:text-amber-300">
            自定义
          </span>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80 mr-1">
          X 轴
        </span>
        {(["step", "epoch", "wallclock"] as XMode[]).map((m) => {
          const active = xMode === m
          const disabled = m === "epoch" && !epochAvailable
          return (
            <button
              key={m}
              type="button"
              disabled={disabled}
              onClick={() => onSelectXMode(m)}
              title={disabled ? "当前后端运行未上报 epoch，使用 step 或时长轴" : undefined}
              className={cn(
                "rounded-[3px] border px-2 py-0.5 text-[10.5px] tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-45",
                active
                  ? "border-primary/45 bg-primary/10 text-foreground"
                  : "border-border/55 bg-background/60 text-muted-foreground hover:text-foreground",
              )}
            >
              {m === "step" ? "step" : m === "epoch" ? "epoch" : "时长"}
            </button>
          )
        })}
      </div>

      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80 mr-1">
          参考
        </span>
        {referenceRun ? (
          isCurrentReference ? (
            <>
              <span className="inline-flex items-center rounded-[3px] border border-emerald-600/40 bg-emerald-600/10 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.16em] text-emerald-700 dark:text-emerald-300">
                当前为基线
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[10.5px]"
                onClick={onClearReference}
              >
                取消
              </Button>
            </>
          ) : (
            <>
              <span
                className="inline-flex items-center rounded-[3px] border border-border/60 bg-background/70 px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground"
                title={referenceRun.jobId}
              >
                {referenceRun.label}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[10.5px]"
                onClick={onPinReference}
                title="将当前任务设为基线"
              >
                替换
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[10.5px]"
                onClick={onClearReference}
              >
                清除
              </Button>
            </>
          )
        ) : (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-[10.5px] gap-1"
            onClick={onPinReference}
            title="把当前任务设为参考基线, 之后其他任务的损失图会叠加它的曲线"
          >
            <Eye className="size-3" />
            设为基线
          </Button>
        )}
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        <PanelToggle
          label="阶段时间线"
          value={panels.showStageTimeline}
          onClick={() => onTogglePanel("showStageTimeline")}
        />
        <PanelToggle
          label="指标细分"
          value={panels.showMetricGrid}
          onClick={() => onTogglePanel("showMetricGrid")}
        />
        <PanelToggle
          label="检查点回放"
          value={panels.showCheckpointPlayback}
          onClick={() => onTogglePanel("showCheckpointPlayback")}
        />
      </div>
    </div>
  )
}

function PanelToggle({
  label,
  value,
  onClick,
}: {
  label: string
  value: boolean
  onClick: () => void
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant="ghost"
      className={cn(
        "h-6 gap-1 px-2 text-[10.5px] tracking-wide",
        value ? "text-foreground" : "text-muted-foreground/70",
      )}
      onClick={onClick}
    >
      {value ? (
        <ChevronUp className="size-3" />
      ) : (
        <ChevronDown className="size-3" />
      )}
      {label}
    </Button>
  )
}
