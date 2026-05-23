/**
 * AnalysisWorkbench — single-job analytical view.
 *
 * Layout (top to bottom):
 *   1. KPI strip   — six headline metrics, one row, no card chrome.
 *   2. Loss panel  — main chart, full width. The visual focus.
 *   3. Secondary   — two-column grid: "LR + iter time + samples/sec"
 *                    (left, dual axis) and "GPU util + VRAM% + temp"
 *                    (right, dual axis).
 *   4. Tabs        — series stats / metrics table / samples / AI
 *                    analysis. Only one is rendered at a time so the
 *                    page never feels like an information avalanche.
 */
import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { ChevronDown, ChevronUp, Eye, Radio } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type JobSummary } from "@/lib/api"
import {
  LossChart,
  type ChartBand,
  type ChartMarker,
  type LossSeries,
} from "../../jobs/components/loss-chart"
import { TERMINAL_STATES } from "../../jobs/utils"
import { AnalysisKpiStrip } from "./analysis-kpi-strip"
import { CheckpointPlayback } from "./checkpoint-playback"
import { DiagnosisBanner } from "./diagnosis-banner"
import { EffectivenessPanel } from "./effectiveness-panel"
import { rollingQuartiles, type BandPoint } from "./loss-stats"
import { MetricGrid } from "./metric-grid"
import { analyseChangepoints } from "./pelt"
import { StageTimeline } from "./stage-timeline"
import {
  defaultPanels,
  inferDefaultMode,
  loadPanelOverrides,
  loadViewMode,
  savePanelOverrides,
  saveViewMode,
  type PanelState,
  type ViewMode,
} from "./view-mode"
import {
  formatXTick,
  loadXMode,
  saveXMode,
  xMapper,
  xModeLabel,
  type XMode,
} from "./x-axis-mode"
import {
  loadReferenceRun,
  saveReferenceRun,
  type ReferenceRun,
} from "./reference-run"
import { AICard } from "../panels/ai-card"
import { MetricsTable } from "../panels/metrics-table"
import { SamplesGallery } from "../panels/samples-gallery"
import { SeriesStatsCard } from "../panels/series-stats"

type BottomTabKey = "stats" | "table" | "samples" | "ai"

const EMA_ALPHA = 0.1

export function AnalysisWorkbench({
  job,
  fallbackTotalSteps,
}: {
  job: JobSummary
  fallbackTotalSteps: number | null
}) {
  const isTerminal = TERMINAL_STATES.has(job.state)
  const metrics = useQuery({
    queryKey: ["job-metrics", job.id],
    queryFn: () => api.getJobMetrics(job.id),
    refetchInterval: isTerminal ? false : 4000,
    staleTime: 2_000,
  })
  const files = useQuery({
    queryKey: ["job-files", job.id],
    queryFn: () => api.getJobFiles(job.id),
    refetchInterval: isTerminal ? false : 8000,
    staleTime: 4_000,
  })

  // View-mode (live / postmortem / custom). Defaults derive from the
  // job's terminal-ness; an explicit user toggle pins the choice.
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    const stored = loadViewMode(job.id)
    return stored ?? inferDefaultMode(isTerminal)
  })
  // Re-derive when the job transitions terminal mid-session — but only
  // if the user hasn't explicitly chosen a mode this session.
  useEffect(() => {
    const stored = loadViewMode(job.id)
    if (stored) return
    setViewMode(inferDefaultMode(isTerminal))
  }, [isTerminal, job.id])
  const [panels, setPanels] = useState<PanelState>(() => {
    const overrides = loadPanelOverrides(job.id) ?? {}
    const base = defaultPanels(loadViewMode(job.id) ?? inferDefaultMode(isTerminal))
    return { ...base, ...overrides }
  })
  function selectMode(next: ViewMode) {
    setViewMode(next)
    saveViewMode(job.id, next)
    if (next !== "custom") {
      const base = defaultPanels(next)
      setPanels(base)
      savePanelOverrides(job.id, {})
    }
  }
  function togglePanel(key: keyof PanelState) {
    setPanels((prev) => {
      const next = { ...prev, [key]: !prev[key] }
      // Any manual flip switches the user into custom mode so future
      // job-state transitions don't stomp their choice.
      if (viewMode !== "custom") {
        setViewMode("custom")
        saveViewMode(job.id, "custom")
      }
      const overrides: Partial<PanelState> = { [key]: next[key] }
      // Merge with any prior overrides so toggling A then B keeps both.
      savePanelOverrides(job.id, {
        ...(loadPanelOverrides(job.id) ?? {}),
        ...overrides,
      })
      return next
    })
  }

  // X-axis mode (step / epoch / wallclock) — persists per-job in
  // sessionStorage so the user's choice survives navigating away.
  const [xMode, setXMode] = useState<XMode>(() => loadXMode(job.id))
  function selectXMode(next: XMode) {
    setXMode(next)
    saveXMode(job.id, next)
  }

  // Reference-run overlay — pinned globally in localStorage. Listens
  // for the storage event so a "set as reference" toggle elsewhere
  // (or in another tab) propagates without a full page reload.
  const [referenceRun, setReferenceRun] = useState<ReferenceRun | null>(() =>
    loadReferenceRun(),
  )
  useEffect(() => {
    const onChange = () => setReferenceRun(loadReferenceRun())
    window.addEventListener("lorahub:reference-run-changed", onChange)
    window.addEventListener("storage", onChange)
    return () => {
      window.removeEventListener("lorahub:reference-run-changed", onChange)
      window.removeEventListener("storage", onChange)
    }
  }, [])
  const isCurrentReference = referenceRun?.jobId === job.id
  const referenceMetrics = useQuery({
    queryKey: ["job-metrics", referenceRun?.jobId],
    queryFn: () => api.getJobMetrics(referenceRun!.jobId),
    enabled: !!referenceRun && !isCurrentReference,
    staleTime: 60_000,
    retry: false,
  })
  function pinAsReference() {
    saveReferenceRun({
      jobId: job.id,
      label: job.id.slice(-8),
    })
    setReferenceRun({ jobId: job.id, label: job.id.slice(-8) })
  }
  function clearReference() {
    saveReferenceRun(null)
    setReferenceRun(null)
  }
  const aiCache = useQuery({
    queryKey: ["job-analysis", job.id],
    queryFn: () => api.getJobAnalysis(job.id),
    refetchInterval: false,
    staleTime: 30_000,
  })

  const lossSeries: LossSeries[] = useMemo(() => {
    const allLossPoints = (metrics.data?.loss ?? []).filter(
      (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
        typeof p.loss === "number" && Number.isFinite(p.loss),
    )
    const map = xMapper(xMode, metrics.data ?? null)
    // Maintain a step → mapped-X table so all derived series (median,
    // EMA, val) use the same X coordinate as the raw series.
    const stepToX = new Map<number, number>()
    for (const p of allLossPoints) {
      stepToX.set(p.step, map(p))
    }
    const trainPoints = allLossPoints.map((p) => ({
      step: stepToX.get(p.step) ?? map(p),
      loss: p.loss,
    }))
    const out: LossSeries[] = []
    if (trainPoints.length > 0) {
      // Use rolling median as the primary line — diffusion training loss
      // is high-variance noise around a slowly-moving signal, so a
      // single raw line over-emphasises step-to-step jitter. Raw stays
      // visible as a faint dashed reference; the IQR band conveys
      // dispersion without forcing the user to read variance from
      // chart wiggle.
      const robust =
        trainPoints.length >= 8 ? rollingQuartiles(trainPoints) : null
      if (robust) {
        out.push({
          id: `${job.id}-train-median`,
          label: "训练 loss · 中位数",
          color: "var(--chart-1)",
          points: robust.median,
        })
        out.push({
          id: `${job.id}-train-raw`,
          label: "原始采样",
          color: "var(--chart-1)",
          dashed: true,
          points: trainPoints,
        })
      } else {
        out.push({
          id: `${job.id}-train`,
          label: "训练 loss",
          color: "var(--chart-1)",
          points: trainPoints,
        })
      }
      if (trainPoints.length >= 6) {
        let acc = trainPoints[0].loss
        const ema = trainPoints.map((p) => {
          acc = EMA_ALPHA * p.loss + (1 - EMA_ALPHA) * acc
          return { step: p.step, loss: acc }
        })
        out.push({
          id: `${job.id}-train-ema`,
          label: `EMA α=${EMA_ALPHA}`,
          color: "var(--chart-1)",
          dashed: true,
          points: ema,
        })
      }
    }
    const valPoints = (metrics.data?.val_loss ?? []).filter(
      (p): p is { epoch: number; val_loss: number; step?: number | null; ts: number } =>
        typeof p.val_loss === "number" && Number.isFinite(p.val_loss),
    )
    if (valPoints.length > 0) {
      const epochToStep = new Map<number, number>()
      for (const p of metrics.data?.loss ?? []) {
        if (typeof p.epoch === "number" && typeof p.step === "number") {
          epochToStep.set(p.epoch, p.step)
        }
      }
      const lastTrainStep = allLossPoints.length
        ? allLossPoints[allLossPoints.length - 1].step
        : 0
      const mapped = valPoints.map((p) => {
        const stepHint =
          typeof p.step === "number"
            ? p.step
            : (epochToStep.get(p.epoch) ?? (lastTrainStep || p.epoch))
        const x = map({
          step: stepHint,
          epoch: p.epoch,
          ts: p.ts,
        })
        return { step: x, loss: p.val_loss }
      })
      out.push({
        id: `${job.id}-val`,
        label: "验证 loss",
        color: "var(--chart-2)",
        points: mapped,
      })
    }
    // Reference-run overlay — drawn as a faint dashed line behind the
    // primary series so users can compare "is this run beating my best
    // baseline" without leaving the page.
    if (
      referenceRun &&
      referenceRun.jobId !== job.id &&
      referenceMetrics.data
    ) {
      const refMap = xMapper(xMode, referenceMetrics.data)
      const refPoints = (referenceMetrics.data.loss ?? [])
        .filter(
          (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
            typeof p.loss === "number" && Number.isFinite(p.loss),
        )
        .map((p) => ({ step: refMap(p), loss: p.loss }))
      if (refPoints.length > 0) {
        out.push({
          id: `${job.id}-ref`,
          label: `参考 ${referenceRun.label}`,
          color: "color-mix(in oklch, var(--muted-foreground) 80%, transparent)",
          dashed: true,
          points: refPoints,
        })
      }
    }
    return out
  }, [metrics.data, job.id, xMode, referenceRun, referenceMetrics.data])

  const checkpointMarkers: ChartMarker[] = useMemo(() => {
    const map = xMapper(xMode, metrics.data ?? null)
    return (metrics.data?.checkpoints ?? [])
      .filter((c): c is { path: string; step: number; ts: number } =>
        typeof c.step === "number" && Number.isFinite(c.step),
      )
      .map((c) => ({
        step: map(c),
        label: c.path,
        color: "var(--chart-3)",
      }))
  }, [metrics.data, xMode])

  // IQR band for the training loss — drawn behind the median line so
  // users can read dispersion at a glance instead of guessing it from
  // the noise envelope of the raw curve.
  const lossBands: ChartBand[] = useMemo(() => {
    const points = (metrics.data?.loss ?? []).filter(
      (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
        typeof p.loss === "number" && Number.isFinite(p.loss),
    )
    if (points.length < 8) return []
    const map = xMapper(xMode, metrics.data ?? null)
    const series: BandPoint[] = rollingQuartiles(
      points.map((p) => ({ step: map(p), loss: p.loss })),
    ).band
    return [
      {
        id: "train-iqr",
        color: "color-mix(in oklch, var(--chart-1) 18%, transparent)",
        points: series,
      },
    ]
  }, [metrics.data, xMode])

  // PELT changepoint analysis — used by the stage timeline below the
  // KPI strip and also overlaid as cyan dashed lines on the loss chart.
  // PELT runs on raw step-axis losses (segment cost is identical under
  // X transforms); we then translate the resulting changepoint X values
  // through the mapper for the markers.
  const changepoints = useMemo(() => {
    const points = (metrics.data?.loss ?? [])
      .filter(
        (p): p is { step: number; loss: number; ts: number } =>
          typeof p.loss === "number" && Number.isFinite(p.loss),
      )
      .map((p) => ({ step: p.step, loss: p.loss }))
    return analyseChangepoints(points)
  }, [metrics.data])

  const allMarkers: ChartMarker[] = useMemo(() => {
    const map = xMapper(xMode, metrics.data ?? null)
    // Reproduce the loss timestamp for each changepoint step so the
    // mapper has wallclock data to work with.
    const stepToSample = new Map<
      number,
      { step: number; epoch?: number | null; ts: number }
    >()
    for (const p of metrics.data?.loss ?? []) {
      if (typeof p.step === "number")
        stepToSample.set(p.step, { step: p.step, epoch: p.epoch, ts: p.ts })
    }
    const out: ChartMarker[] = [...checkpointMarkers]
    for (const s of changepoints.changepointSteps) {
      const sample = stepToSample.get(s) ?? { step: s, ts: 0 }
      out.push({ step: map(sample), color: "var(--chart-2)" })
    }
    return out
  }, [checkpointMarkers, changepoints.changepointSteps, metrics.data, xMode])

  const totalPoints = metrics.data?.loss?.length ?? 0
  // Estimated training progress in [0..1] for context-aware tone
  // selection in the effectiveness panel. Prefers the trainer-reported
  // total_steps over the config-derived fallback.
  const progress: number | null = useMemo(() => {
    const lossArr = metrics.data?.loss ?? []
    const lastStep = lossArr.length
      ? lossArr[lossArr.length - 1].step
      : null
    const totalSteps =
      typeof metrics.data?.total_steps === "number" &&
      metrics.data.total_steps > 0
        ? metrics.data.total_steps
        : (fallbackTotalSteps ?? null)
    if (
      lastStep == null ||
      totalSteps == null ||
      totalSteps <= 0 ||
      !Number.isFinite(lastStep) ||
      !Number.isFinite(totalSteps)
    ) {
      return null
    }
    return Math.max(0, Math.min(1, lastStep / totalSteps))
  }, [metrics.data, fallbackTotalSteps])
  const overfit = metrics.data?.overfit_signal

  // Bottom tabs default + counts (used in tab labels).
  const [bottomTab, setBottomTab] = useState<BottomTabKey>("stats")
  const samplesCount = files.data?.samples?.length ?? 0
  const tableRowCount = totalPoints
  const seriesCount = lossSeries.length
  const aiState: "ready" | "missing" = aiCache.data?.analysis ? "ready" : "missing"

  return (
    <div className="flex flex-col min-h-0">
      <AnalysisKpiStrip job={job} fallbackTotalSteps={fallbackTotalSteps} />

      <div className="px-7 py-4 space-y-4">
        {/* View-mode switcher: live / postmortem / custom. Mode picks
            sensible defaults for which heavy panels are open by
            default; manual toggles flip into custom and persist. */}
        <ViewModeSwitcher
          mode={viewMode}
          panels={panels}
          isTerminal={isTerminal}
          xMode={xMode}
          referenceRun={referenceRun}
          isCurrentReference={isCurrentReference}
          onSelectMode={selectMode}
          onTogglePanel={togglePanel}
          onSelectXMode={selectXMode}
          onPinReference={pinAsReference}
          onClearReference={clearReference}
        />

        {/* Diagnosis banner — most urgent signal first. Auto-runs once
            on mount; the backend route is read-only so re-running it
            on demand is also safe. */}
        <DiagnosisBanner jobId={job.id} />

        {/* Effectiveness insights — convergence / stability / overfit /
            stage. Sits above the chart so users get the verdict before
            squinting at the curve. */}
        <EffectivenessPanel
          metrics={metrics.data ?? null}
          progress={progress}
        />

        {/* PELT-derived stage timeline — colour-coded segments + slope
            annotation. Render only when at least one changepoint was
            found, otherwise the user just sees a single bar that
            contributes no information. */}
        {panels.showStageTimeline && changepoints.segments.length > 1 && (
          <StageTimeline
            segments={changepoints.segments}
            changepointSteps={changepoints.changepointSteps}
          />
        )}

        {/* Loss panel — visual focus of the page (train + val + EMA). */}
        <Card
          className="analysis-fade-in-stagger"
          style={{ ["--stagger-delay" as string]: "320ms" }}
        >
          <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
            <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              损失曲线 · 综合视图
            </CardTitle>
            <span className="text-[10px] text-muted-foreground/70">
              {metrics.isLoading
                ? "加载中…"
                : metrics.isError
                  ? "加载失败"
                  : `${totalPoints} 个采样点${
                      totalPoints > 1000 ? "（已下采样到 1000）" : ""
                    }${
                      checkpointMarkers.length > 0
                        ? ` · ${checkpointMarkers.length} 个检查点`
                        : ""
                    }${
                      changepoints.changepointSteps.length > 0
                        ? ` · ${changepoints.changepointSteps.length} 个变点`
                        : ""
                    }`}
            </span>
          </CardHeader>
          <CardContent className="p-3.5">
            <LossChart
              series={lossSeries}
              bands={lossBands}
              overfitSignal={overfit ?? null}
              markers={allMarkers}
              xLabel={xModeLabel(xMode)}
              xTickFormat={formatXTick(xMode)}
              persistKey={`${job.id}.${xMode}`}
            />
          </CardContent>
        </Card>

        {/* Detailed metrics — TensorBoard-style breakdown. One small
            card per scalar (loss/raw, loss/ema, loss/epoch_avg,
            schedule/learning_rate, throughput.*) so the user can
            isolate any signal without scanning a dense legend. */}
        {panels.showMetricGrid && (
          <div
            className="analysis-fade-in-stagger"
            style={{ ["--stagger-delay" as string]: "400ms" }}
          >
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2 px-0.5">
              指标细分
            </div>
            <MetricGrid metrics={metrics.data ?? null} jobId={job.id} xMode={xMode} />
          </div>
        )}

        {/* Checkpoint playback — same prompt × same seed across every
            checkpoint. The single most reliable visual quality signal
            for diffusion fine-tuning. */}
        {panels.showCheckpointPlayback && (
          <CheckpointPlayback
            jobId={job.id}
            samples={files.data?.samples ?? []}
            loading={files.isLoading}
          />
        )}

        {/* Bottom tabs — stats / table / samples / AI. */}
        <Tabs
          value={bottomTab}
          onValueChange={(v) => setBottomTab(v as BottomTabKey)}
        >
          <div className="border-y border-border/60 bg-muted/40 px-3.5 py-1.5 rounded-t-[6px]">
            <TabsList variant="line" className="gap-3">
              <TabsTrigger value="stats" className="text-[11.5px]">
                序列统计{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {seriesCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="table" className="text-[11.5px]">
                指标表格{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {tableRowCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="samples" className="text-[11.5px]">
                样本{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {samplesCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="ai" className="text-[11.5px]">
                AI 分析{" "}
                <span
                  className={
                    aiState === "ready"
                      ? "ml-1 text-emerald-600 dark:text-emerald-400"
                      : "ml-1 text-muted-foreground/80"
                  }
                >
                  {aiState === "ready" ? "✓" : "未生成"}
                </span>
              </TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="stats" className="m-0">
            <SeriesStatsCard
              series={lossSeries}
              metrics={metrics.data ?? null}
            />
          </TabsContent>
          <TabsContent value="table" className="m-0">
            <MetricsTable
              loss={metrics.data?.loss ?? []}
              valLoss={metrics.data?.val_loss ?? []}
              loading={metrics.isLoading}
            />
          </TabsContent>
          <TabsContent value="samples" className="m-0">
            <SamplesGallery
              jobId={job.id}
              samples={files.data?.samples ?? []}
              loading={files.isLoading}
            />
          </TabsContent>
          <TabsContent value="ai" className="m-0">
            <AICard
              jobId={job.id}
              cached={aiCache.data?.analysis ?? null}
              loading={aiCache.isLoading}
              canRun={isTerminal || (metrics.data?.loss?.length ?? 0) > 0}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

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

function ViewModeSwitcher({
  mode,
  panels,
  isTerminal,
  xMode,
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
          return (
            <button
              key={m}
              type="button"
              onClick={() => onSelectXMode(m)}
              className={cn(
                "rounded-[3px] border px-2 py-0.5 text-[10.5px] tracking-wide transition-colors",
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
