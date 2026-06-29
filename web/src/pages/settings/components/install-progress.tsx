import { Check, CircleDashed, Loader2, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import type { BackendId, BootstrapEvent } from "@/lib/api"

export type StepDef = { id: string; label: string; match: RegExp }

export const STEP_PLANS: Record<BackendId, StepDef[]> = {
  kohya: [
    { id: "clone", label: "克隆仓库", match: /^clone\s+kohya-ss\//i },
    { id: "venv", label: "创建虚拟环境", match: /^(uv\s+venv\b|create\s+venv)/i },
    { id: "torch", label: "安装 PyTorch", match: /^install\s+torch/i },
    {
      id: "requirements",
      label: "安装 kohya requirements",
      match: /kohya\s+requirements/i,
    },
    { id: "xformers", label: "安装 xformers", match: /^install\s+xformers/i },
  ],
  "diffusion-pipe": [
    { id: "clone", label: "克隆仓库", match: /^clone\s+tdrussell\//i },
    { id: "venv", label: "创建虚拟环境", match: /^(uv\s+venv\b|create\s+venv)/i },
    { id: "torch", label: "安装 PyTorch", match: /^install\s+torch/i },
    {
      id: "requirements",
      label: "安装 diffusion-pipe requirements",
      match: /diffusion-pipe\s+requirements/i,
    },
    { id: "deepspeed", label: "安装 DeepSpeed", match: /^install\s+deepspeed/i },
  ],
  anima_lora: [
    { id: "sync", label: "uv sync (创建 .venv + 装依赖)", match: /^uv\s+sync/i },
  ],
  ai_toolkit: [
    { id: "venv", label: "创建虚拟环境", match: /^(uv\s+venv\b|create\s+venv)/i },
    { id: "torch", label: "安装 PyTorch", match: /^install\s+torch/i },
    { id: "requirements", label: "安装 ai-toolkit requirements", match: /ai-toolkit\s+requirements/i },
  ],
}

export type StepState = "pending" | "running" | "succeeded" | "failed"

export function isTerminalStatus(status: string): boolean {
  return ["succeeded", "failed", "canceled", "interrupted"].includes(status)
}

export function isRetryableStatus(status: string): boolean {
  return ["failed", "canceled", "interrupted"].includes(status)
}

export function computeStepStates(
  plan: StepDef[],
  events: BootstrapEvent[],
  status: string,
): { states: StepState[]; current: number } {
  const states: StepState[] = plan.map(() => "pending")
  let current = -1
  for (const ev of events) {
    if (ev.level !== "info") continue
    const idx = plan.findIndex((s) => s.match.test(ev.message))
    if (idx < 0) continue
    if (current >= 0 && current < plan.length) states[current] = "succeeded"
    current = idx
    states[idx] = "running"
  }
  if (current >= 0) {
    if (status === "succeeded") {
      for (let i = 0; i < states.length; i += 1) {
        if (states[i] === "pending" || states[i] === "running") {
          states[i] = "succeeded"
        }
      }
    } else if (isRetryableStatus(status)) {
      states[current] = "failed"
    }
  } else if (status === "succeeded") {
    for (let i = 0; i < states.length; i += 1) states[i] = "succeeded"
  }
  return { states, current }
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "succeeded")
    return <Check className="size-3.5 text-emerald-600 dark:text-emerald-400" />
  if (state === "running")
    return <Loader2 className="size-3.5 animate-spin text-amber-600 dark:text-amber-400" />
  if (state === "failed") return <XCircle className="size-3.5 text-destructive" />
  return <CircleDashed className="size-3.5 text-muted-foreground/60" />
}

export function StepList({
  plan,
  states,
}: {
  plan: StepDef[]
  states: StepState[]
}) {
  return (
    <ol className="rounded-[4px] border border-border/60 bg-muted/30 divide-y divide-border/40">
      {plan.map((s, i) => {
        const state = states[i]
        return (
          <li
            key={s.id}
            className={cn(
              "px-3 py-2 flex items-center gap-3 text-xs",
              state === "running" && "bg-amber-500/5",
              state === "failed" && "bg-destructive/5",
              state === "succeeded" && "text-muted-foreground",
            )}
          >
            <StepIcon state={state} />
            <span className="flex-1">{s.label}</span>
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
              {state === "pending" ? "等待" : state === "running" ? "进行中" : state === "succeeded" ? "完成" : "失败"}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

export function ProgressBar({
  done,
  total,
  failed,
}: {
  done: number
  total: number
  failed: boolean
}) {
  const pct = total > 0 ? Math.min(100, Math.max(0, (done / total) * 100)) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>
          已完成 {done} / {total}
        </span>
        <span className="font-mono tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      <div className="shiro-progress-track h-1.5">
        <div
          className={cn(
            "shiro-progress-fill",
            failed ? "bg-destructive" : "bg-emerald-500",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "succeeded"
      ? "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/5"
      : status === "failed"
        ? "text-destructive border-destructive/40 bg-destructive/5"
        : status === "running"
          ? "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/5"
          : "text-muted-foreground border-border/60 bg-muted/30"
  const label =
    status === "succeeded"
      ? "已完成"
      : status === "failed"
        ? "失败"
        : status === "running"
          ? "运行中"
          : "空闲"
  return (
    <span
      className={cn(
        "px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] rounded-[3px] border font-mono",
        tone,
      )}
    >
      {label}
    </span>
  )
}

export function EventLog({ events }: { events: BootstrapEvent[] }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-zinc-950 dark:bg-zinc-900 max-h-72 overflow-y-auto">
      <ol className="divide-y divide-zinc-800/60 font-mono text-[11px]">
        {events.map((ev, idx) => (
          <li
            key={`${ev.ts}-${idx}`}
            className={cn(
              "px-3 py-1.5 flex items-start gap-2",
              ev.level === "error"
                ? "text-red-400 bg-red-950/40 border-l-2 border-l-red-500"
                : ev.level === "done"
                  ? "text-emerald-400"
                  : "text-zinc-100",
            )}
          >
            <span className="text-[9px] uppercase tracking-[0.18em] text-zinc-500 w-12 shrink-0 pt-0.5">
              {ev.level}
            </span>
            <span className="break-all">{ev.message}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
