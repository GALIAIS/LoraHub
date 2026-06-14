import { useEffect, useMemo, useRef } from "react"
import {
  AlertOctagon,
  CheckCircle2,
  Database,
  FileImage,
  Flag,
  ImageIcon,
  Layers,
  Play,
  Save,
  Search,
  Square,
  XCircle,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { TrainingEvent } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"
import { parseAnsi } from "./ansi"
import {
  CONTEXT_AFTER,
  CONTEXT_BEFORE,
  EVENT_LEVEL,
  cachePhaseLabel,
  fmtClock,
  milestoneTitle,
  renderInlineSummary,
  toneFor,
  type Milestone,
  type MilestoneKind,
} from "./event-timeline-model"

interface KindStyle {
  icon: typeof Play
  label: string
  badge: string
  text: string
}

const KIND_STYLES: Record<MilestoneKind, KindStyle> = {
  spawn: {
    icon: Play,
    label: "进程启动",
    badge: "bg-blue-500/20 text-blue-500 border border-blue-500/40",
    text: "text-blue-600 dark:text-blue-400",
  },
  cache_phase: {
    icon: Database,
    label: "缓存阶段",
    badge: "bg-amber-500/20 text-amber-600 border border-amber-500/40",
    text: "text-amber-700 dark:text-amber-300",
  },
  epoch: {
    icon: Layers,
    label: "回合",
    badge: "bg-violet-500/20 text-violet-600 border border-violet-500/40",
    text: "text-violet-700 dark:text-violet-300",
  },
  validation: {
    icon: Search,
    label: "验证",
    badge: "bg-cyan-500/20 text-cyan-600 border border-cyan-500/40",
    text: "text-cyan-700 dark:text-cyan-300",
  },
  checkpoint: {
    icon: Save,
    label: "检查点",
    badge: "bg-emerald-500/20 text-emerald-600 border border-emerald-500/40",
    text: "text-emerald-700 dark:text-emerald-300",
  },
  sample: {
    icon: FileImage,
    label: "样本",
    badge: "bg-fuchsia-500/20 text-fuchsia-600 border border-fuchsia-500/40",
    text: "text-fuchsia-700 dark:text-fuchsia-300",
  },
  error: {
    icon: XCircle,
    label: "错误",
    badge: "bg-red-500/20 text-red-500 border border-red-500/50",
    text: "text-red-600 dark:text-red-400",
  },
  oom: {
    icon: AlertOctagon,
    label: "显存溢出",
    badge: "bg-orange-500/20 text-orange-500 border border-orange-500/50",
    text: "text-orange-700 dark:text-orange-300",
  },
  done: {
    icon: Flag,
    label: "结束",
    badge: "bg-foreground/15 text-foreground border border-foreground/30",
    text: "text-foreground",
  },
}

export function TimelineRail({
  milestones,
  selectedId,
  onSelect,
  filter,
}: {
  milestones: Milestone[]
  selectedId: string | null
  onSelect: (id: string) => void
  filter: Set<MilestoneKind>
}) {
  const filtered = useMemo(
    () => milestones.filter((m) => filter.has(m.kind)),
    [milestones, filter],
  )
  const tailRef = useRef<HTMLLIElement | null>(null)

  useEffect(() => {
    if (!tailRef.current) return
    tailRef.current.scrollIntoView({ block: "end" })
  }, [filtered.length])

  if (filtered.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-muted-foreground">
        正在等待事件…
      </div>
    )
  }

  return (
    <ScrollArea className="h-full">
      <ol className="relative px-3 py-3 space-y-1.5">
        <span
          aria-hidden
          className="absolute top-0 bottom-0 left-[calc(0.75rem+10px)] w-px bg-border/50"
        />
        {filtered.map((m, i) => {
          const style = KIND_STYLES[m.kind]
          const Icon = style.icon
          const active = selectedId === m.id
          const isLast = i === filtered.length - 1
          return (
            <li
              key={m.id}
              ref={isLast ? tailRef : undefined}
              className="relative"
            >
              <button
                type="button"
                onClick={() => onSelect(m.id)}
                className={cn(
                  "group flex w-full items-start gap-2.5 rounded-[5px] py-1 pr-2 pl-0 text-left transition-colors",
                  active ? "bg-primary/10" : "hover:bg-muted/40",
                )}
              >
                <span
                  className={cn(
                    "z-10 mt-[2px] flex size-5 shrink-0 items-center justify-center rounded-full",
                    style.badge,
                    active && "ring-2 ring-primary/40",
                  )}
                  aria-hidden
                >
                  <Icon className="size-3" strokeWidth={2.4} />
                </span>
                <span className="min-w-0 flex-1">
                  <span
                    className={cn(
                      "block text-[12px] font-medium leading-tight truncate",
                      style.text,
                    )}
                  >
                    {milestoneTitle(m)}
                  </span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {fmtClock(m.ts)}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </ScrollArea>
  )
}

export function DetailPane({
  milestone,
  events,
  jobId,
  fallbackTotalSteps,
}: {
  milestone: Milestone | null
  events: TrainingEvent[]
  jobId: string | null
  fallbackTotalSteps: number | null
}) {
  if (!milestone) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
        从左侧选择一个里程碑以查看详情。
      </div>
    )
  }
  const style = KIND_STYLES[milestone.kind]
  const Icon = style.icon
  const begin = Math.max(0, milestone.anchorIndex - CONTEXT_BEFORE)
  const end = Math.min(events.length, milestone.anchorIndex + CONTEXT_AFTER + 1)
  const ctx = events.slice(begin, end)

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border/60 bg-muted/30 px-5 py-3">
        <div className="flex items-start gap-3">
          <span
            className={cn(
              "flex size-9 shrink-0 items-center justify-center rounded-full",
              style.badge,
            )}
            aria-hidden
          >
            <Icon className="size-4" strokeWidth={2.4} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              {style.label} · {fmtClock(milestone.ts)}
            </div>
            <div className={cn("text-[15px] font-semibold leading-snug", style.text)}>
              {milestoneTitle(milestone)}
            </div>
          </div>
        </div>
        <DetailBody milestone={milestone} jobId={jobId} />
      </header>
      <div className="min-h-0 flex-1">
        <ScrollArea className="h-full">
          <div className="px-5 py-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                上下文（前 {milestone.anchorIndex - begin} · 后 {end - milestone.anchorIndex - 1} 条）
              </span>
            </div>
            <ContextLog
              ctx={ctx}
              anchorIndexInCtx={milestone.anchorIndex - begin}
              fallbackTotalSteps={fallbackTotalSteps}
            />
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

function DetailBody({
  milestone,
  jobId,
}: {
  milestone: Milestone
  jobId: string | null
}) {
  switch (milestone.kind) {
    case "cache_phase": {
      const done = (milestone.data.done as number) ?? 0
      const total = (milestone.data.total as number) ?? 0
      const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0
      return (
        <div className="mt-3 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] tabular-nums text-muted-foreground">
            <span>{cachePhaseLabel(String(milestone.data.phase ?? ""))}</span>
            <span>
              {done} / {total || "?"} ({pct.toFixed(0)}%)
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-amber-500 transition-[width] duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )
    }
    case "validation": {
      const v = milestone.data.val_loss
      const epoch = milestone.data.epoch
      const step = milestone.data.step
      return (
        <div className="mt-2 grid grid-cols-3 gap-3 text-[12px]">
          <DetailKV label="验证 loss" value={typeof v === "number" ? v.toFixed(4) : "—"} />
          <DetailKV label="回合" value={epoch != null ? String(epoch) : "—"} />
          <DetailKV label="步" value={step != null ? String(step) : "—"} />
        </div>
      )
    }
    case "checkpoint":
      return (
        <div className="mt-2">
          <DetailKV
            label="检查点路径"
            value={String(milestone.data.path ?? "—")}
            mono
          />
          {milestone.data.step != null && (
            <DetailKV label="步" value={String(milestone.data.step)} />
          )}
        </div>
      )
    case "sample": {
      const path = String(milestone.data.path ?? "")
      return (
        <div className="mt-2 space-y-2">
          <DetailKV label="文件" value={path} mono />
          {jobId && path && (
            <a
              href={`/api/jobs/${jobId}/files/raw?path=${encodeURIComponent(path)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="block max-w-[220px]"
            >
              <img
                src={`/api/jobs/${jobId}/files/raw?path=${encodeURIComponent(path)}`}
                alt={path}
                className="rounded-[4px] border border-border/40 bg-muted/30 object-contain"
                loading="lazy"
              />
            </a>
          )}
          {!path && (
            <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
              <ImageIcon className="size-3.5" /> 未提供路径
            </div>
          )}
        </div>
      )
    }
    case "error":
    case "oom": {
      const msg = String(
        milestone.data.error ?? milestone.data.message ?? milestone.data.traceback ?? "",
      )
      const tb = milestone.data.traceback
      return (
        <div className="mt-2 space-y-2">
          {msg && (
            <pre className="whitespace-pre-wrap rounded-[4px] border border-red-500/40 bg-red-500/5 p-2 font-mono text-[11.5px] text-red-600 dark:text-red-400">
              {msg}
            </pre>
          )}
          {typeof tb === "string" && tb !== msg && (
            <details className="rounded-[4px] border border-border/40 bg-muted/20 p-2 text-[11px]">
              <summary className="cursor-pointer text-muted-foreground">完整 traceback</summary>
              <pre className="mt-2 max-h-[280px] overflow-auto whitespace-pre-wrap font-mono">
                {tb}
              </pre>
            </details>
          )}
        </div>
      )
    }
    case "done": {
      const rc = milestone.data.returncode
      const dur = milestone.data.duration_s
      const ok = rc === 0
      return (
        <div className="mt-2 grid grid-cols-2 gap-3 text-[12px]">
          <DetailKV
            label="返回码"
            value={
              <span
                className={cn(
                  "inline-flex items-center gap-1",
                  ok
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-red-600 dark:text-red-400",
                )}
              >
                {ok ? (
                  <CheckCircle2 className="size-3" />
                ) : (
                  <Square className="size-3" />
                )}
                {String(rc ?? "—")}
              </span>
            }
          />
          <DetailKV
            label="用时"
            value={typeof dur === "number" ? `${dur.toFixed(1)}s` : "—"}
          />
        </div>
      )
    }
    case "epoch":
      return (
        <div className="mt-2 grid grid-cols-2 gap-3 text-[12px]">
          <DetailKV label="完成回合" value={String(milestone.data.epoch ?? "—")} />
          <DetailKV
            label="总回合"
            value={String(milestone.data.total_epochs ?? "—")}
          />
        </div>
      )
    default:
      return null
  }
}

function DetailKV({
  label,
  value,
  mono,
}: {
  label: string
  value: React.ReactNode
  mono?: boolean
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "mt-0.5 break-all text-[12px]",
          mono && "font-mono text-foreground/90",
        )}
      >
        {value}
      </div>
    </div>
  )
}

function ContextLog({
  ctx,
  anchorIndexInCtx,
  fallbackTotalSteps,
}: {
  ctx: TrainingEvent[]
  anchorIndexInCtx: number
  fallbackTotalSteps: number | null
}) {
  if (ctx.length === 0) {
    return (
      <div className="text-[12px] text-muted-foreground">无上下文事件。</div>
    )
  }
  return (
    <ul className="font-mono text-[11.5px] leading-[1.6]">
      {ctx.map((e, i) => {
        const time = fmtClock(e.timestamp)
        const isAnchor = i === anchorIndexInCtx
        const tone = toneFor(e)
        const summary = renderInlineSummary(e, fallbackTotalSteps)
        const chunks = parseAnsi(summary)
        return (
          <li
            key={`${i}-${e.timestamp}-${e.type}`}
            className={cn(
              "flex gap-2 border-l-2 px-2 py-[2px]",
              isAnchor
                ? "border-l-primary bg-primary/5"
                : "border-l-transparent hover:bg-muted/30",
            )}
          >
            <span className="shrink-0 tabular-nums text-muted-foreground/70">
              [{time}]
            </span>
            <span className="w-[88px] shrink-0 text-muted-foreground/80">
              {EVENT_LEVEL[e.type as keyof typeof EVENT_LEVEL] ?? e.type}
            </span>
            <span className={cn("min-w-0 flex-1 whitespace-pre-wrap break-all", tone)}>
              {chunks.length === 0 ? summary : chunks.map((c, ci) => (
                <span key={ci} className={c.className}>{c.text}</span>
              ))}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
