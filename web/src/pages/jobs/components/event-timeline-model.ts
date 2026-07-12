import type { TrainingEvent } from "@/lib/api"

export const CONTEXT_BEFORE = 6
export const CONTEXT_AFTER = 14

export type MilestoneKind =
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

export interface Milestone {
  kind: MilestoneKind
  ts: number
  anchorIndex: number
  id: string
  data: Record<string, unknown>
}

export function buildMilestones(events: TrainingEvent[]): Milestone[] {
  const out: Milestone[] = []
  let spawnEmitted = false
  let lastSeenStep: number | null = null
  let lastSeenEpoch: number | null = null
  const openCachePhases = new Map<
    string,
    { last: CachePhase; firstIndex: number; lastIndex: number }
  >()

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

    if (e.type === "step" && typeof p.step === "number") {
      lastSeenStep = p.step
    }
    if (typeof p.epoch === "number") lastSeenEpoch = p.epoch

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
      if (last.total > 0 && last.done >= last.total) flushCachePhase(phase)
      return
    }

    if (openCachePhases.size > 0) {
      for (const name of Array.from(openCachePhases.keys())) flushCachePhase(name)
    }

    if (e.type === "epoch_start" || e.type === "epoch_end") {
      out.push({
        kind: "epoch",
        ts: e.timestamp,
        anchorIndex: idx,
        id: `epoch-${idx}`,
        data: { ...p, phase: e.type },
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
  })

  for (const name of Array.from(openCachePhases.keys())) flushCachePhase(name)
  return out
}

export function fmtClock(ts: number): string {
  const d = new Date(ts * 1000)
  return `${String(d.getHours()).padStart(2, "0")}:${String(
    d.getMinutes(),
  ).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`
}

export function milestoneTitle(m: Milestone): string {
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
      return `第 ${m.data.epoch ?? "?"}/${m.data.total_epochs ?? "?"} 回合${m.data.phase === "epoch_start" ? "开始" : "结束"}`
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

export function cachePhaseLabel(name: string): string {
  if (name === "latents") return "潜空间缓存"
  if (name === "text_encoder") return "文本编码缓存"
  return name
}

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
  const path = typeof enriched.path === "string" ? (enriched.path as string) : null
  if (path) {
    const epochMatch = path.match(/epoch[-_]?(\d+)/i)
    if (epochMatch) enriched.epoch = Number(epochMatch[1])
    const stepMatch = path.match(/step[-_]?(\d+)/i)
    if (stepMatch) enriched.step = Number(stepMatch[1])
  }
  return enriched
}

export const EVENT_LEVEL: Record<string, string> = {
  step: "STEP",
  epoch_start: "EPOCH",
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

export function toneFor(e: TrainingEvent): string {
  const lvl = String((e.payload as Record<string, unknown>)?.level ?? "")
  if (e.type === "error" || e.type === "oom" || /ERROR|CRITICAL|FATAL/.test(lvl))
    return "text-red-600 dark:text-red-400"
  if (/WARN/.test(lvl)) return "text-amber-700 dark:text-amber-300"
  if (e.type === "step") return "text-cyan-700 dark:text-cyan-400"
  if (e.type === "checkpoint_saved")
    return "text-emerald-700 dark:text-emerald-400"
  if (e.type === "sample_ready")
    return "text-fuchsia-700 dark:text-fuchsia-400"
  if (e.type === "epoch_start" || e.type === "epoch_end")
    return "text-violet-700 dark:text-violet-400"
  if (e.type === "validation") return "text-cyan-700 dark:text-cyan-400"
  if (e.type === "preview_unavailable")
    return "text-amber-700 dark:text-amber-300"
  if (e.type === "done") return "text-emerald-700 dark:text-emerald-400"
  return "text-foreground/80"
}

export function renderInlineSummary(
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
    case "epoch_start":
      return `第 ${p.epoch ?? "?"}/${p.total_epochs ?? "?"} 回合开始`
    case "validation":
      return `验证 loss=${
        typeof p.val_loss === "number" ? (p.val_loss as number).toFixed(4) : "?"
      }`
    case "checkpoint_saved":
      return `保存检查点 → ${p.path ?? ""}`
    case "sample_ready":
      return `生成样本 → ${p.path ?? ""}`
    case "preview_unavailable": {
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

export const ALL_KINDS: MilestoneKind[] = [
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

export const FILTER_CHIPS: Array<{
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
