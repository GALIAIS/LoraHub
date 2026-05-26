/**
 * RunSummaryCard
 *
 * Single glance health snapshot for the active job. Sits directly under
 * the job-detail header so the user doesn't have to open the analysis
 * tab to know "is this run healthy and what knobs am I looking at".
 *
 * Sections:
 *   1. Progress: current step / total steps / ETA / wall time.
 *   2. Loss: starting value, latest value, percent drop, convergence
 *      shape, train-vs-val gap if validation is present.
 *   3. Hyper-params snapshot: lr / rank / alpha / network_dropout /
 *      num_repeats / batch x grad_accum / epochs.
 *
 * No backend calls beyond the metrics + config_snapshot the parent
 * already fetches. Stateless apart from a memoised derive of summary
 * stats.
 */
import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  ArrowDownRight,
  BarChart3,
  ChevronDown,
  ExternalLink,
  Hourglass,
  Zap,
} from "lucide-react"
import { Card, CardContent, CardTitle } from "@/components/ui/card"
import type { JobDetail, JobMetricsResponse } from "@/lib/api"
import { fmtDuration } from "../utils"
import { cn } from "@/lib/utils"

// localStorage key — collapsed state should persist across reloads so
// the user only flips the toggle once per session.
const COLLAPSED_KEY = "lorahub.jobs.runSummary.collapsed"

interface Props {
  job: JobDetail | undefined
  metrics: JobMetricsResponse | undefined
  fallbackTotalSteps: number | null
}

export function RunSummaryCard({ job, metrics, fallbackTotalSteps }: Props) {
  const navigate = useNavigate()
  const summary = useMemo(
    () => deriveSummary(job, metrics, fallbackTotalSteps),
    [job, metrics, fallbackTotalSteps],
  )

  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    const stored = window.localStorage.getItem(COLLAPSED_KEY)
    // Default to collapsed when nothing's stored yet — most of the
    // time the user wants to glance at the inline digest and only
    // expand when something looks off.
    if (stored === null) return true
    return stored === "1"
  })
  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0")
  }, [collapsed])

  if (!summary) return null

  return (
    <Card>
      <div
        className={cn(
          "w-full flex items-center justify-between gap-3",
          "py-2.5 px-4 bg-muted/40 hover:bg-muted/60 transition-colors",
          collapsed ? "border-b-0" : "border-b border-border/60",
        )}
      >
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          aria-expanded={!collapsed}
          className="flex items-center gap-3 min-w-0 text-left flex-1 hover:text-foreground transition-colors"
        >
          <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground shrink-0">
            训练健康摘要
          </CardTitle>
          {collapsed && <CollapsedSnapshot summary={summary} />}
        </button>
        <div className="flex items-center gap-1 shrink-0">
          {(() => {
            const url =
              typeof job?.metadata?.wandb_run_url === "string"
                ? (job.metadata.wandb_run_url as string)
                : null
            if (!url) return null
            return (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                title="在 wandb.ai 打开本次 run"
                className="inline-flex items-center gap-1 rounded-[3px] border border-transparent px-1.5 py-0.5 text-[11px] text-muted-foreground hover:border-border/60 hover:bg-background/60 hover:text-foreground"
              >
                <ExternalLink className="size-3" /> W&amp;B
              </a>
            )
          })()}
          {job?.id && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                navigate(`/analysis/${job.id}`)
              }}
              title="在训练分析中打开（loss / AI 分析）"
              className="inline-flex items-center gap-1 rounded-[3px] border border-transparent px-1.5 py-0.5 text-[11px] text-muted-foreground hover:border-border/60 hover:bg-background/60 hover:text-foreground"
            >
              <BarChart3 className="size-3" /> 分析
            </button>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "展开摘要" : "折叠摘要"}
            className="p-1 text-muted-foreground/80 hover:text-foreground"
          >
            <ChevronDown
              className={cn(
                "size-3.5 transition-transform",
                collapsed ? "-rotate-90" : "rotate-0",
              )}
              aria-hidden
            />
          </button>
        </div>
      </div>
      {!collapsed && (
        <CardContent className="p-3 grid gap-3 grid-cols-1 md:grid-cols-3">
          <ProgressBlock summary={summary} />
          <LossBlock summary={summary} />
          <HparamsBlock summary={summary} />
        </CardContent>
      )}
    </Card>
  )
}

// One-line digest shown inline with the header when the card is
// collapsed. Picks the four numbers a user is most likely to glance
// at to decide whether to expand: progress, latest loss, drop %, and
// convergence trend label.
function CollapsedSnapshot({ summary }: { summary: Summary }) {
  const parts: React.ReactNode[] = []
  if (summary.step != null) {
    parts.push(
      <span key="step" className="tabular-nums">
        {summary.step}
        {summary.totalSteps != null ? `/${summary.totalSteps}` : ""}
        {summary.percent != null ? (
          <span className="text-muted-foreground/70">
            {" "}
            ({summary.percent.toFixed(0)}%)
          </span>
        ) : null}
      </span>,
    )
  }
  if (summary.lossLatest != null) {
    parts.push(
      <span key="loss" className="tabular-nums">
        loss {summary.lossLatest.toFixed(4)}
      </span>,
    )
  }
  if (summary.lossDropPct != null) {
    parts.push(
      <span
        key="drop"
        className={cn(
          "tabular-nums",
          summary.lossDropPct > 0
            ? "text-emerald-600 dark:text-emerald-400"
            : "text-amber-700 dark:text-amber-400",
        )}
      >
        {summary.lossDropPct > 0 ? "↓" : "↑"}{" "}
        {Math.abs(summary.lossDropPct).toFixed(1)}%
      </span>,
    )
  }
  if (summary.trend !== "unknown") {
    parts.push(
      <span key="trend" className={cn("text-[11px]", trendTone(summary.trend))}>
        {trendLabel(summary.trend)}
      </span>,
    )
  }
  if (parts.length === 0) {
    return (
      <span className="text-[11px] text-muted-foreground/70">
        等待第一组指标…
      </span>
    )
  }
  return (
    <span className="flex items-center gap-3 text-[11px] text-foreground/85 truncate">
      {parts.map((p, i) => (
        <span key={i} className="flex items-center gap-3">
          {p}
          {i < parts.length - 1 && (
            <span className="text-border" aria-hidden>
              ·
            </span>
          )}
        </span>
      ))}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function ProgressBlock({ summary }: { summary: Summary }) {
  const { step, totalSteps, percent, etaSec, wallSec } = summary
  return (
    <div className="space-y-1.5">
      <SectionLabel icon={<Hourglass className="size-3" />} label="训练进度" />
      <div className="flex items-baseline gap-1.5">
        <span className="text-[20px] font-semibold tabular-nums leading-none">
          {step ?? "—"}
        </span>
        <span className="text-[11px] text-muted-foreground tabular-nums">
          / {totalSteps ?? "?"} 步
          {percent != null && (
            <span className="ml-1.5 text-foreground/80">{percent.toFixed(1)}%</span>
          )}
        </span>
      </div>
      {percent != null && (
        <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.min(percent, 100)}%` }}
          />
        </div>
      )}
      <KV label="已用时" value={fmtDuration(wallSec)} />
      <KV
        label="预计剩余"
        value={etaSec != null ? fmtDuration(etaSec) : "—"}
      />
    </div>
  )
}

function LossBlock({ summary }: { summary: Summary }) {
  const { lossStart, lossLatest, lossDropPct, trend, valGap, overfit } = summary
  return (
    <div className="space-y-1.5">
      <SectionLabel icon={<ArrowDownRight className="size-3" />} label="损失趋势" />
      {lossLatest != null ? (
        <>
          <div className="flex items-baseline gap-1.5">
            <span className="text-[20px] font-semibold tabular-nums leading-none">
              {lossLatest.toFixed(4)}
            </span>
            {lossStart != null && lossDropPct != null && (
              <span
                className={cn(
                  "text-[11px] tabular-nums",
                  lossDropPct > 0
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-amber-700 dark:text-amber-400",
                )}
              >
                {lossDropPct > 0 ? "↓" : "↑"}{" "}
                {Math.abs(lossDropPct).toFixed(1)}% 自起始
              </span>
            )}
          </div>
          {lossStart != null && (
            <div className="text-[11px] text-muted-foreground tabular-nums">
              起始 {lossStart.toFixed(4)} → 当前 {lossLatest.toFixed(4)}
            </div>
          )}
          <KV label="收敛趋势" value={trendLabel(trend)} tone={trendTone(trend)} />
          {valGap != null && (
            <KV
              label="train↔val"
              value={`+${valGap.toFixed(4)}`}
              tone={
                overfit === "warn"
                  ? "text-amber-700 dark:text-amber-400"
                  : overfit === "danger"
                    ? "text-red-600 dark:text-red-400"
                    : "text-foreground/80"
              }
            />
          )}
        </>
      ) : (
        <div className="text-[11px] text-muted-foreground py-1">
          尚未收到 loss 数据点
        </div>
      )}
    </div>
  )
}

function HparamsBlock({ summary }: { summary: Summary }) {
  const h = summary.hparams
  return (
    <div className="space-y-1.5">
      <SectionLabel icon={<Zap className="size-3" />} label="关键超参" />
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
        <KV label="lr" value={h.lr ?? "—"} mono />
        <KV label="epochs" value={h.epochs ?? "—"} mono />
        <KV label="rank" value={h.rank ?? "—"} mono />
        <KV label="alpha" value={h.alpha ?? "—"} mono />
        <KV label="dropout" value={h.dropout ?? "—"} mono />
        <KV label="repeats" value={h.numRepeats ?? "—"} mono />
        <KV label="batch×accum" value={h.batchAccum ?? "—"} mono />
        <KV label="optimizer" value={h.optimizer ?? "—"} mono />
      </div>
    </div>
  )
}

function SectionLabel({
  icon,
  label,
}: {
  icon: React.ReactNode
  label: string
}) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80">
      {icon}
      {label}
    </div>
  )
}

function KV({
  label,
  value,
  tone,
  mono,
}: {
  label: string
  value: React.ReactNode
  tone?: string
  mono?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 text-[11px]">
      <span className="text-muted-foreground/80 shrink-0">{label}</span>
      <span
        className={cn(
          "min-w-0 truncate text-right",
          mono && "font-mono tabular-nums",
          tone ?? "text-foreground/85",
        )}
      >
        {value}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

type Trend = "improving" | "flat" | "diverging" | "unknown"
type Overfit = null | "warn" | "danger"

interface Summary {
  step: number | null
  totalSteps: number | null
  percent: number | null
  etaSec: number | null
  wallSec: number | null
  lossStart: number | null
  lossLatest: number | null
  lossDropPct: number | null
  trend: Trend
  valGap: number | null
  overfit: Overfit
  hparams: {
    lr: string | null
    rank: string | null
    alpha: string | null
    dropout: string | null
    numRepeats: string | null
    batchAccum: string | null
    epochs: string | null
    optimizer: string | null
  }
}

function deriveSummary(
  job: JobDetail | undefined,
  metrics: JobMetricsResponse | undefined,
  fallbackTotalSteps: number | null,
): Summary | null {
  const cfg = (job?.config_snapshot ?? {}) as Record<string, unknown>
  if (Object.keys(cfg).length === 0 && !metrics) return null

  // Loss series (cleaned of non-finite entries)
  const points = (metrics?.loss ?? []).filter(
    (p): p is { step: number; loss: number; ts: number } =>
      typeof p.loss === "number" && Number.isFinite(p.loss),
  )

  const lossStart = points.length > 0 ? points[0].loss : null
  const lossLatest = points.length > 0 ? points[points.length - 1].loss : null
  const lossDropPct =
    lossStart != null && lossLatest != null && lossStart > 0
      ? ((lossStart - lossLatest) / lossStart) * 100
      : null

  // Trend: compare last 20% vs prior 20% of points to classify shape.
  const trend = classifyTrend(points)

  // Validation gap (latest val_loss - latest train loss in the same epoch).
  const vals = metrics?.val_loss ?? []
  let valGap: number | null = null
  let overfit: Overfit = null
  if (vals.length > 0 && lossLatest != null) {
    const latestVal = vals[vals.length - 1].val_loss
    if (typeof latestVal === "number" && Number.isFinite(latestVal)) {
      valGap = latestVal - lossLatest
      if (valGap > 0.5 * Math.max(lossLatest, 1e-3)) overfit = "danger"
      else if (valGap > 0.2 * Math.max(lossLatest, 1e-3)) overfit = "warn"
    }
  }

  // Progress
  const step = lossLatest != null ? points[points.length - 1].step : null
  const totalSteps = pickTotalSteps(metrics, cfg, fallbackTotalSteps)
  const percent =
    step != null && totalSteps != null && totalSteps > 0
      ? (step / totalSteps) * 100
      : null
  const wallSec =
    metrics?.first_step_ts != null && metrics?.last_step_ts != null
      ? metrics.last_step_ts - metrics.first_step_ts
      : null
  const etaSec =
    wallSec != null && step != null && totalSteps != null && step > 0
      ? (wallSec / step) * (totalSteps - step)
      : null

  // Hparams snapshot — accept both camelCase (new YAMLs) and snake_case
  // (legacy snapshots persisted before the schema migration). Pull each
  // field through a tolerant helper.
  const network = pluckObj(cfg, "network") ?? {}
  const optimizer = pluckObj(cfg, "optimizer") ?? {}
  const schedule = pluckObj(cfg, "schedule") ?? {}
  const dataset = pluckObj(cfg, "dataset") ?? {}
  const backend = pluckObj(cfg, "backend") ?? {}
  const backendType = stringOrNull(pluck(backend, "type"))
  // anima_lora reads learning rate from backend.animaLora.learningRate;
  // kohya / diffusion-pipe read it from optimizer.lr.unet. Mirror the
  // compiler's source-of-truth field per backend so the displayed value
  // matches what the trainer actually consumes.
  const animaLora = pluckObj(backend, "animaLora") ?? pluckObj(backend, "anima_lora") ?? {}
  const lr =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "learningRate", "learning_rate")
      : pluck(pluckObj(optimizer, "lr") ?? {}, "unet")
  const optimizerType =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "optimizerType", "optimizer_type")
      : pluck(optimizer, "type")
  const batchSize = pluckOne(schedule, "batchSize", "batch_size")
  const gradAccum = pluckOne(schedule, "gradAccum", "grad_accum")

  // anima_lora reads rank/alpha/dropout from backend.animaLora.*
  // (networkDim / networkAlpha / lora.networkDropout); kohya / dp
  // read from the top-level network section. Mirror the lr / optimizer
  // routing above so the summary always reflects what train.py actually
  // consumed.
  const animaLoraLora = pluckObj(animaLora, "lora") ?? {}
  const rank =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "networkDim", "network_dim")
      : pluck(network, "rank")
  const alpha =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "networkAlpha", "network_alpha")
      : pluck(network, "alpha")
  const networkDropout =
    backendType === "anima_lora"
      ? pluckOne(animaLoraLora, "networkDropout", "network_dropout")
      : pluckOne(network, "networkDropout", "network_dropout")
  const hparams = {
    lr: typeof lr === "number" ? scientific(lr) : null,
    rank: stringOrNull(rank),
    alpha: stringOrNull(alpha),
    dropout: numberOrNull(networkDropout),
    numRepeats: stringOrNull(pluckOne(dataset, "numRepeats", "num_repeats")),
    batchAccum:
      batchSize != null && gradAccum != null
        ? `${batchSize}×${gradAccum}`
        : stringOrNull(batchSize),
    epochs: stringOrNull(pluck(schedule, "epochs")),
    optimizer: stringOrNull(optimizerType),
  }

  return {
    step,
    totalSteps,
    percent,
    etaSec,
    wallSec,
    lossStart,
    lossLatest,
    lossDropPct,
    trend,
    valGap,
    overfit,
    hparams,
  }
}

function classifyTrend(
  points: Array<{ step: number; loss: number }>,
): Trend {
  if (points.length < 6) return "unknown"
  const tail = points.slice(Math.floor(points.length * 0.8))
  const head = points.slice(
    Math.floor(points.length * 0.6),
    Math.floor(points.length * 0.8),
  )
  if (tail.length === 0 || head.length === 0) return "unknown"
  const tailAvg = tail.reduce((s, p) => s + p.loss, 0) / tail.length
  const headAvg = head.reduce((s, p) => s + p.loss, 0) / head.length
  const delta = (tailAvg - headAvg) / Math.max(headAvg, 1e-6)
  if (delta < -0.02) return "improving"
  if (delta > 0.05) return "diverging"
  return "flat"
}

function trendLabel(t: Trend): string {
  switch (t) {
    case "improving":
      return "下降中"
    case "flat":
      return "平台期"
    case "diverging":
      return "上升发散"
    default:
      return "—"
  }
}

function trendTone(t: Trend): string {
  switch (t) {
    case "improving":
      return "text-emerald-600 dark:text-emerald-400"
    case "flat":
      return "text-amber-700 dark:text-amber-400"
    case "diverging":
      return "text-red-600 dark:text-red-400"
    default:
      return "text-foreground/85"
  }
}

function pluck(obj: unknown, key: string): unknown {
  if (obj && typeof obj === "object" && key in (obj as object)) {
    return (obj as Record<string, unknown>)[key]
  }
  return null
}

function pluckObj(obj: unknown, key: string): Record<string, unknown> | null {
  const v = pluck(obj, key)
  return v && typeof v === "object" ? (v as Record<string, unknown>) : null
}

function pluckOne(obj: unknown, ...keys: string[]): unknown {
  if (!obj || typeof obj !== "object") return undefined
  const o = obj as Record<string, unknown>
  for (const k of keys) {
    if (k in o && o[k] != null) return o[k]
  }
  return undefined
}

function pickTotalSteps(
  metrics: JobMetricsResponse | undefined,
  cfg: Record<string, unknown>,
  fallback: number | null,
): number | null {
  // Trainer-reported total wins — it's the only number that survives
  // mid-run schedule changes (warmup steps rolling in, sample_ratio
  // truncating the dataset, …). Falls back to schedule.maxSteps in the
  // recipe, then to the config-derived estimate (epochs × repeats ×
  // images / (batch × accum)) the parent computes off the dataset
  // scan. This is the same priority order overview-tab uses, so the
  // three tabs now agree.
  const fromMetrics = metrics?.total_steps
  if (typeof fromMetrics === "number" && fromMetrics > 0) return fromMetrics
  const sched = pluckObj(cfg, "schedule") ?? {}
  const explicit = pluckOne(sched, "maxSteps", "max_steps")
  if (typeof explicit === "number" && explicit > 0) return explicit
  return fallback
}

function scientific(n: number): string {
  if (!Number.isFinite(n)) return "—"
  if (n === 0) return "0"
  const abs = Math.abs(n)
  if (abs >= 1e-3 && abs < 1e3) return n.toString()
  return n.toExponential(2)
}

function stringOrNull(v: unknown): string | null {
  if (v == null) return null
  return String(v)
}

function numberOrNull(v: unknown): string | null {
  if (typeof v !== "number" || !Number.isFinite(v)) return null
  return v.toFixed(3).replace(/\.?0+$/, "")
}
