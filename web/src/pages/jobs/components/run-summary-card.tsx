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
import {
  deriveSummary,
  trendLabel,
  trendTone,
  type Summary,
} from "./run-summary-model"

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
        <div className="shiro-progress-track h-1 w-full border-0 bg-muted">
          <div
            className="shiro-progress-fill bg-primary"
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
