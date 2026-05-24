/**
 * Event timeline (A 方案): two-pane layout for the events tab.
 *
 * Left rail compresses the entire run into a vertical stack of milestone
 * markers (spawn / cache progress bars / per-epoch / validation /
 * checkpoint / sample / error / oom / done). Step/log/gpu_sample noise is
 * elided from the rail but still indexed so the right-hand context panel
 * can replay the surrounding lines around any milestone the user clicks.
 *
 * The right pane has three modes:
 *   - milestone selected: header + structured details + ±N log context
 *   - filter "all" / "log" with no selection: virtualised raw stream
 *   - error tab: jump-list of every error / oom event
 *
 * No backend changes needed; we feed off the same events array
 * useJobStream already streams.
 */
import { useEffect, useMemo, useRef, useState } from "react"
import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Database,
  FileImage,
  Flag,
  ImageIcon,
  Layers,
  Play,
  Save,
  Search,
  ServerCog,
  Square,
  XCircle,
  Zap,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { TrainingEvent } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { parseAnsi } from "./ansi"

// Number of events to render around the focused milestone in the
// detail pane. Tuned so a 4090 epoch's worth of step lines doesn't
// bury the sample/checkpoint that triggered them.
const CONTEXT_BEFORE = 6
const CONTEXT_AFTER = 14

// ---------------------------------------------------------------------------
// Milestone classifier
// ---------------------------------------------------------------------------

type MilestoneKind =
  | "spawn"
  | "cache_phase"
  | "epoch"
  | "validation"
  | "checkpoint"
  | "sample"
  | "error"
  | "oom"
  | "done"

interface CachePhase {
  phase: string
  done: number
  total: number
  ts: number
}

interface Milestone {
  kind: MilestoneKind
  // Timestamp the milestone is anchored at. We use payload-derived ts when
  // available (e.g. the ts of the LAST cache_progress event for the phase),
  // otherwise the source event's timestamp.
  ts: number
  // Index into the events[] this milestone "lives at". For aggregated
  // milestones (cache_phase) it's the index of the LAST event that
  // contributed; the context window scrolls around this.
  anchorIndex: number
  // Stable id for React keying + selection persistence across refetches.
  id: string
  // Free-form payload presented in the detail header.
  data: Record<string, unknown>
}

/**
 * Walk the events list once, emitting milestones for the timeline rail.
 * Aggregates streams of cache_progress events so the user sees one row
 * per phase with a progress bar instead of dozens of incremental rows.
 */
function buildMilestones(events: TrainingEvent[]): Milestone[] {
  const out: Milestone[] = []
  let spawnEmitted = false
  // Track the latest (step, epoch) we've observed in any event; checkpoint
  // / validation / sample events from dp don't carry step in the payload,
  // so we hang the most recent train step on them retroactively. Without
  // this the rail shows "保存检查点 · 步 ?" forever.
  let lastSeenStep: number | null = null
  let lastSeenEpoch: number | null = null
  // Open cache phase aggregator: collapses N progress events into one
  // milestone keyed by `phase`. We close the milestone (commit it) when
  // the next non-cache event arrives or when done==total.
  const openCachePhases = new Map<string, { last: CachePhase; firstIndex: number; lastIndex: number }>()

  function flushCachePhase(name: string) {
    const open = openCachePhases.get(name)
    if (!open) return
    out.push({
      kind: "cache_phase",
      ts: open.last.ts,
      anchorIndex: open.lastIndex,
      id: `cache-${name}-${open.firstIndex}`,
      data: { ...open.last },
    })
    openCachePhases.delete(name)
  }

  events.forEach((e, idx) => {
    const p = e.payload ?? {}

    // Update the running step / epoch trackers eagerly so milestones
    // emitted later in the same iteration can read from them.
    if (e.type === "step" && typeof p.step === "number") {
      lastSeenStep = p.step
    }
    if (typeof p.epoch === "number") lastSeenEpoch = p.epoch

    // First non-meta event for the run becomes the "spawn" anchor — we
    // synthesise it from whatever came first so the user has somewhere
    // to click to see the launch banner / argv.
    if (!spawnEmitted) {
      spawnEmitted = true
      out.push({
        kind: "spawn",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `spawn-${idx}`,
        data: { firstEvent: e },
      })
    }

    if (e.type === "cache_progress") {
      const phase = String(p.phase ?? "cache")
      const open = openCachePhases.get(phase)
      const last: CachePhase = {
        phase,
        done: typeof p.done === "number" ? p.done : (open?.last.done ?? 0),
        total: typeof p.total === "number" ? p.total : (open?.last.total ?? 0),
        ts: e.timestamp,
      }
      if (open) {
        open.last = last
        open.lastIndex = idx
      } else {
        openCachePhases.set(phase, { last, firstIndex: idx, lastIndex: idx })
      }
      // Emit immediately on completion so it doesn't dangle until end.
      if (last.total > 0 && last.done >= last.total) flushCachePhase(phase)
      return
    }

    // Any other event terminates open cache phases.
    if (openCachePhases.size > 0) {
      for (const name of Array.from(openCachePhases.keys())) flushCachePhase(name)
    }

    if (e.type === "epoch_end") {
      out.push({
        kind: "epoch",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `epoch-${idx}`,
        data: { ...p },
      })
    } else if (e.type === "validation") {
      out.push({
        kind: "validation",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `val-${idx}`,
        data: enrichWithStepEpoch(p, lastSeenStep, lastSeenEpoch),
      })
    } else if (e.type === "checkpoint_saved") {
      out.push({
        kind: "checkpoint",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `ckpt-${idx}`,
        data: enrichCheckpoint(p, lastSeenStep, lastSeenEpoch),
      })
    } else if (e.type === "sample_ready") {
      out.push({
        kind: "sample",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `sample-${idx}`,
        data: enrichWithStepEpoch(p, lastSeenStep, lastSeenEpoch),
      })
    } else if (e.type === "oom") {
      out.push({
        kind: "oom",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `oom-${idx}`,
        data: { ...p },
      })
    } else if (e.type === "error") {
      out.push({
        kind: "error",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `err-${idx}`,
        data: { ...p },
      })
    } else if (e.type === "done") {
      out.push({
        kind: "done",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `done-${idx}`,
        data: { ...p },
      })
    }
    // step / log / gpu_sample don't make the rail.
  })

  // Drain any cache phases still open at the tail.
  for (const name of Array.from(openCachePhases.keys())) flushCachePhase(name)
  return out
}

// ---------------------------------------------------------------------------
// Visual tokens per milestone kind
// ---------------------------------------------------------------------------

interface KindStyle {
  icon: typeof Play
  label: string
  badge: string // tailwind classes for the dot badge
  text: string  // text color for the title
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

function fmtClock(ts: number): string {
  const d = new Date(ts * 1000)
  return `${String(d.getHours()).padStart(2, "0")}:${String(
    d.getMinutes(),
  ).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`
}

// ---------------------------------------------------------------------------
// Title / summary string for a milestone (used in rail + header)
// ---------------------------------------------------------------------------

function milestoneTitle(m: Milestone): string {
  switch (m.kind) {
    case "spawn":
      return "训练进程已启动"
    case "cache_phase": {
      const phase = String(m.data.phase ?? "")
      const done = (m.data.done as number) ?? 0
      const total = (m.data.total as number) ?? 0
      return `${cachePhaseLabel(phase)} ${done}/${total || "?"}`
    }
    case "epoch":
      return `第 ${m.data.epoch ?? "?"}/${m.data.total_epochs ?? "?"} 回合结束`
    case "validation": {
      const v = m.data.val_loss
      return `验证 loss = ${typeof v === "number" ? v.toFixed(4) : "?"}`
    }
    case "checkpoint":
      return `保存检查点 · 步 ${m.data.step ?? "?"}`
    case "sample":
      return "样本生成"
    case "error":
      return String(m.data.error ?? m.data.message ?? "训练错误")
    case "oom":
      return "CUDA 显存溢出"
    case "done": {
      const rc = m.data.returncode
      const dur = m.data.duration_s
      return `进程结束 · 返回码 ${rc ?? "?"} · 用时 ${
        typeof dur === "number" ? `${dur.toFixed(1)}s` : "?"
      }`
    }
  }
}

function cachePhaseLabel(name: string): string {
  if (name === "latents") return "潜空间缓存"
  if (name === "text_encoder") return "文本编码缓存"
  return name
}

// dp's checkpoint / validation / sample events don't include a step number,
// but the user still expects to know "which step does this belong to".
// Two strategies: (1) parse the path tail (`epoch5/...`, `step120.png`),
// (2) fall back to the most recent step we observed in the event stream.
function enrichWithStepEpoch(
  payload: Record<string, unknown>,
  lastSeenStep: number | null,
  lastSeenEpoch: number | null,
): Record<string, unknown> {
  const enriched: Record<string, unknown> = { ...payload }
  if (enriched.step == null && lastSeenStep != null) enriched.step = lastSeenStep
  if (enriched.epoch == null && lastSeenEpoch != null)
    enriched.epoch = lastSeenEpoch
  return enriched
}

function enrichCheckpoint(
  payload: Record<string, unknown>,
  lastSeenStep: number | null,
  lastSeenEpoch: number | null,
): Record<string, unknown> {
  const enriched = enrichWithStepEpoch(payload, lastSeenStep, lastSeenEpoch)
  // Path-derived hints take priority over the live counter — dp writes
  // `epoch5/` directories so the path is more authoritative than a step
  // counter that's still ticking.
  const path = typeof enriched.path === "string" ? (enriched.path as string) : null
  if (path) {
    const epochMatch = path.match(/epoch[-_]?(\d+)/i)
    if (epochMatch) enriched.epoch = Number(epochMatch[1])
    const stepMatch = path.match(/step[-_]?(\d+)/i)
    if (stepMatch) enriched.step = Number(stepMatch[1])
  }
  return enriched
}

// ---------------------------------------------------------------------------
// Pane: timeline rail
// ---------------------------------------------------------------------------

function TimelineRail({
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

  // Auto-scroll the rail when a new last item appears (live tailing).
  // ``block: "end"`` (not ``"nearest"``) so the very first paint —
  // when the rail mounts on a back-to-events tab switch and the tail
  // is offscreen — actually lands at the bottom instead of relying on
  // "nearest" deciding nothing needs to move.
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
        {/* Vertical guide line */}
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
                  active
                    ? "bg-primary/10"
                    : "hover:bg-muted/40",
                )}
              >
                {/* Badge */}
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

// ---------------------------------------------------------------------------
// Pane: detail (header + structured body + context window)
// ---------------------------------------------------------------------------

function DetailPane({
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

// ---------------------------------------------------------------------------
// Context log: ±N events surrounding a milestone, with anchor highlight
// ---------------------------------------------------------------------------

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

const EVENT_LEVEL: Record<string, string> = {
  step: "STEP",
  epoch_end: "EPOCH",
  validation: "VAL",
  checkpoint_saved: "CKPT",
  sample_ready: "SAMPLE",
  cache_progress: "CACHE",
  oom: "OOM",
  gpu_sample: "GPU",
  preview_unavailable: "PREVIEW",
  done: "DONE",
  log: "LOG",
  error: "ERROR",
}

function toneFor(e: TrainingEvent): string {
  const lvl = String((e.payload as Record<string, unknown>)?.level ?? "")
  if (e.type === "error" || e.type === "oom" || /ERROR|CRITICAL|FATAL/.test(lvl))
    return "text-red-600 dark:text-red-400"
  if (/WARN/.test(lvl)) return "text-amber-700 dark:text-amber-300"
  if (e.type === "step") return "text-cyan-700 dark:text-cyan-400"
  if (e.type === "checkpoint_saved")
    return "text-emerald-700 dark:text-emerald-400"
  if (e.type === "sample_ready")
    return "text-fuchsia-700 dark:text-fuchsia-400"
  if (e.type === "epoch_end") return "text-violet-700 dark:text-violet-400"
  if (e.type === "validation") return "text-cyan-700 dark:text-cyan-400"
  if (e.type === "preview_unavailable")
    return "text-amber-700 dark:text-amber-300"
  if (e.type === "done") return "text-emerald-700 dark:text-emerald-400"
  return "text-foreground/80"
}

function renderInlineSummary(
  e: TrainingEvent,
  fallbackTotalSteps: number | null,
): string {
  const p = e.payload ?? {}
  switch (e.type) {
    case "step": {
      const total =
        typeof p.total_steps === "number" && p.total_steps > 0
          ? (p.total_steps as number)
          : fallbackTotalSteps
      const lossStr =
        typeof p.loss === "number" ? ` · loss=${(p.loss as number).toFixed(4)}` : ""
      return `第 ${p.step ?? "?"}/${total ?? "?"} 步${lossStr}`
    }
    case "log":
      return String(p.message ?? "")
    case "cache_progress":
      return `${cachePhaseLabel(String(p.phase ?? ""))} ${p.done ?? "?"}/${p.total ?? "?"}`
    case "gpu_sample":
      return `util=${p.util_percent ?? "—"}% · vram=${p.vram_used_mib ?? "—"}/${p.vram_total_mib ?? "—"}MiB · ${p.temperature_c ?? "—"}°C`
    case "epoch_end":
      return `第 ${p.epoch ?? "?"}/${p.total_epochs ?? "?"} 回合结束`
    case "validation":
      return `验证 loss=${
        typeof p.val_loss === "number" ? (p.val_loss as number).toFixed(4) : "?"
      }`
    case "checkpoint_saved":
      return `保存检查点 → ${p.path ?? ""}`
    case "sample_ready":
      return `生成样本 → ${p.path ?? ""}`
    case "preview_unavailable": {
      // Cut-3 / B5 — payload: { arch, available_backends, reason }
      const arch = String(p.arch ?? "?")
      const reason = String(p.reason ?? "")
      const tail = reason ? ` · ${reason}` : ""
      return `预览暂不可用（arch=${arch}）${tail} · 训练继续，但不会生成预览图。`
    }
    case "oom":
      return String(p.message ?? "CUDA out of memory")
    case "error":
      return String(p.error ?? p.message ?? p.traceback ?? "训练错误")
    case "done":
      return `进程结束 · returncode=${p.returncode ?? "?"} · 用时 ${
        typeof p.duration_s === "number" ? `${(p.duration_s as number).toFixed(1)}s` : "?"
      }`
    default:
      return JSON.stringify(p)
  }
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

const ALL_KINDS: MilestoneKind[] = [
  "spawn",
  "cache_phase",
  "epoch",
  "validation",
  "checkpoint",
  "sample",
  "error",
  "oom",
  "done",
]

const FILTER_CHIPS: Array<{
  id: "all" | "errors" | "milestones"
  label: string
  kinds: Set<MilestoneKind>
}> = [
  {
    id: "all",
    label: "全部",
    kinds: new Set(ALL_KINDS),
  },
  {
    id: "milestones",
    label: "里程碑",
    kinds: new Set([
      "spawn",
      "epoch",
      "checkpoint",
      "sample",
      "validation",
      "done",
    ]),
  },
  {
    id: "errors",
    label: "异常",
    kinds: new Set(["error", "oom"]),
  },
]

export function EventTimeline({
  events,
  jobId,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  jobId: string | null
  fallbackTotalSteps?: number | null
}) {
  const milestones = useMemo(() => buildMilestones(events), [events])
  const [filterId, setFilterId] = useState<(typeof FILTER_CHIPS)[number]["id"]>(
    "all",
  )
  const filter = FILTER_CHIPS.find((f) => f.id === filterId)?.kinds ?? new Set(ALL_KINDS)
  const filteredMilestones = useMemo(
    () => milestones.filter((m) => filter.has(m.kind)),
    [milestones, filter],
  )

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [followLatest, setFollowLatest] = useState(true)
  const [query, setQuery] = useState("")

  // Auto-follow latest milestone unless the user has explicitly clicked one.
  useEffect(() => {
    if (!followLatest) return
    if (filteredMilestones.length === 0) return
    setSelectedId(filteredMilestones[filteredMilestones.length - 1].id)
  }, [filteredMilestones, followLatest])

  const selected = useMemo(
    () => milestones.find((m) => m.id === selectedId) ?? null,
    [milestones, selectedId],
  )

  const counts = useMemo(() => {
    const c = { all: milestones.length, errors: 0, milestones: 0 }
    for (const m of milestones) {
      if (FILTER_CHIPS[1].kinds.has(m.kind)) c.milestones += 1
      if (FILTER_CHIPS[2].kinds.has(m.kind)) c.errors += 1
    }
    return c
  }, [milestones])

  // Search across milestones (title + payload string).
  const railMilestones = useMemo(() => {
    if (!query) return filteredMilestones
    const q = query.toLowerCase()
    return filteredMilestones.filter((m) => {
      const t = milestoneTitle(m).toLowerCase()
      const data = JSON.stringify(m.data).toLowerCase()
      return t.includes(q) || data.includes(q)
    })
  }, [filteredMilestones, query])

  return (
    <div className="grid h-full min-h-0 grid-cols-[300px_1fr] overflow-hidden rounded-[6px] border border-border/60 bg-background">
      {/* LEFT: rail */}
      <div className="flex min-h-0 flex-col border-r border-border/60 bg-muted/20">
        <div className="border-b border-border/60 px-3 py-2">
          <div className="mb-1.5 flex items-center gap-1">
            {FILTER_CHIPS.map((c) => {
              const active = filterId === c.id
              const n =
                c.id === "all"
                  ? counts.all
                  : c.id === "milestones"
                    ? counts.milestones
                    : counts.errors
              return (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => {
                    setFilterId(c.id)
                    setFollowLatest(true)
                  }}
                  className={cn(
                    "flex h-6 items-center gap-1.5 rounded-[4px] border px-2 text-[11px] transition-colors",
                    active
                      ? "border-primary/40 bg-primary/15 text-foreground"
                      : "border-border/50 bg-background/60 text-muted-foreground hover:text-foreground",
                  )}
                >
                  {c.id === "errors" && (
                    <AlertTriangle
                      className={cn(
                        "size-3",
                        n > 0 ? "text-red-500" : "text-muted-foreground",
                      )}
                    />
                  )}
                  {c.id === "milestones" && <ServerCog className="size-3" />}
                  {c.id === "all" && <Zap className="size-3" />}
                  {c.label}
                  <span className="font-mono text-[10px] tabular-nums text-muted-foreground/80">
                    {n}
                  </span>
                </button>
              )
            })}
          </div>
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索里程碑…"
            className="h-7 text-[11px]"
          />
        </div>
        <div className="min-h-0 flex-1">
          <TimelineRail
            milestones={railMilestones}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id)
              setFollowLatest(false)
            }}
            filter={filter}
          />
        </div>
        {!followLatest && (
          <div className="border-t border-border/60 px-3 py-1.5">
            <button
              type="button"
              onClick={() => setFollowLatest(true)}
              className="text-[10px] uppercase tracking-[0.18em] text-primary hover:underline"
            >
              返回最新事件 →
            </button>
          </div>
        )}
      </div>
      {/* RIGHT: detail */}
      <div className="min-h-0">
        <DetailPane
          milestone={selected}
          events={events}
          jobId={jobId}
          fallbackTotalSteps={fallbackTotalSteps}
        />
      </div>
    </div>
  )
}
