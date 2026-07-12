import { cn } from "@/lib/utils"
import type { TrainingEvent } from "@/lib/api"
import { EVENT_TYPE_LABELS } from "../utils"

function renderPayload(e: TrainingEvent, fallbackTotalSteps: number | null): string {
  const p = e.payload
  switch (e.type) {
    case "step": {
      const total =
        typeof p.total_steps === "number" && p.total_steps > 0
          ? (p.total_steps as number)
          : fallbackTotalSteps
      const totalLabel = total ?? "?"
      return `第 ${p.step}/${totalLabel} 步${
        p.loss !== undefined ? ` · 损失 ${(p.loss as number).toFixed(4)}` : ""
      }`
    }
    case "epoch_end":
      return `第 ${p.epoch}/${p.total_epochs ?? "?"} 回合结束`
    case "epoch_start":
      return `第 ${p.epoch}/${p.total_epochs ?? "?"} 回合开始`
    case "checkpoint_saved":
      return String(p.path ?? "")
    case "sample_ready":
      return String(p.path ?? "")
    case "done":
      return `返回码 ${p.returncode} · 用时 ${
        (p.duration_s as number)?.toFixed?.(1) ?? "?"
      }s`
    case "log":
      return String(p.message ?? "")
    case "diagnostic_warning": {
      // Emitted by SubprocessRunner's StreamingDiagnosticWatcher.
      // Field shape mirrors PreflightFinding so the same label /
      // remediation idiom works on both surfaces.
      const cat = String(p.category ?? "")
      const msg = String(p.message ?? "")
      return cat ? `[${cat}] ${msg}` : msg
    }
    default:
      return JSON.stringify(p)
  }
}

export function EventRow({
  event,
  fallbackTotalSteps = null,
}: {
  event: TrainingEvent
  fallbackTotalSteps?: number | null
}) {
  const time = new Date(event.timestamp * 1000).toLocaleTimeString()
  const summary = renderPayload(event, fallbackTotalSteps)
  const tone =
    {
      error: "text-destructive",
      done: "text-emerald-600 dark:text-emerald-400",
      checkpoint_saved: "text-cyan-700 dark:text-cyan-400",
      sample_ready: "text-fuchsia-700 dark:text-fuchsia-400",
      epoch_start: "text-primary",
      epoch_end: "text-primary",
      diagnostic_warning: "text-amber-700 dark:text-amber-400",
    }[event.type] ?? "text-foreground"
  const label = EVENT_TYPE_LABELS[event.type] ?? event.type

  return (
    <li className="px-4 py-1.5 flex gap-3 items-baseline hover:bg-muted/30">
      <span className="text-muted-foreground/60 shrink-0 text-[11px]">
        {time}
      </span>
      <span
        className={cn("shrink-0 w-[120px] text-[11px] tracking-wide", tone)}
      >
        {label}
      </span>
      <span className="text-foreground/80 truncate">{summary}</span>
    </li>
  )
}
