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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type JobDetail, type JobSummary } from "@/lib/api"
import { useAnimeAnalysisMotion } from "@/hooks/use-anime-analysis-motion"
import { LossChart } from "../../jobs/components/loss-chart"
import { MAX_POINTS } from "../../jobs/components/loss-chart-model"
import { TERMINAL_STATES } from "../../jobs/utils"
import {
  buildAllMarkers,
  buildChangepoints,
  buildCheckpointMarkers,
  buildLossBands,
  buildLossSeries,
} from "./analysis-chart-model"
import { AnalysisKpiStrip } from "./analysis-kpi-strip"
import { deriveAnalysisBackendInfo } from "./analysis-backend"
import { BackendContextStrip } from "./backend-context-strip"
import { ViewModeSwitcher } from "./analysis-view-switcher"
import { CheckpointPlayback } from "./checkpoint-playback"
import { DiagnosisBanner } from "./diagnosis-banner"
import { EffectivenessPanel } from "./effectiveness-panel"
import { MetricGrid } from "./metric-grid"
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
import { WandbTab } from "../panels/wandb-tab"

type BottomTabKey = "stats" | "table" | "samples" | "ai"
type ViewKind = "builtin" | "wandb"

export function AnalysisWorkbench({
  job,
  jobDetail,
  fallbackTotalSteps,
}: {
  job: JobSummary
  jobDetail: JobDetail | null
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
  const backendInfo = useMemo(
    () => deriveAnalysisBackendInfo(jobDetail),
    [jobDetail],
  )

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
  const referenceDetail = useQuery({
    queryKey: ["job", referenceRun?.jobId],
    queryFn: () => api.getJob(referenceRun!.jobId),
    enabled: !!referenceRun && !isCurrentReference,
    staleTime: 60_000,
    retry: false,
  })
  const referenceBackendInfo = useMemo(
    () => deriveAnalysisBackendInfo(referenceDetail.data),
    [referenceDetail.data],
  )
  const crossBackendReference =
    !!referenceRun &&
    !isCurrentReference &&
    backendInfo.type != null &&
    referenceBackendInfo.type != null &&
    backendInfo.type !== referenceBackendInfo.type
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

  const lossSeries = useMemo(
    () =>
      buildLossSeries({
        metrics: metrics.data,
        jobId: job.id,
        xMode,
        referenceRun,
        referenceMetrics: referenceMetrics.data,
        backendType: backendInfo.type,
        referenceBackendType: referenceBackendInfo.type,
      }),
    [
      metrics.data,
      job.id,
      xMode,
      referenceRun,
      referenceMetrics.data,
      backendInfo.type,
      referenceBackendInfo.type,
    ],
  )

  const checkpointMarkers = useMemo(
    () => buildCheckpointMarkers(metrics.data, xMode),
    [metrics.data, xMode],
  )

  // IQR band for the training loss — drawn behind the median line so
  // users can read dispersion at a glance instead of guessing it from
  // the noise envelope of the raw curve.
  const lossBands = useMemo(
    () => buildLossBands(metrics.data, xMode),
    [metrics.data, xMode],
  )

  // PELT changepoint analysis — used by the stage timeline below the
  // KPI strip and also overlaid as cyan dashed lines on the loss chart.
  // PELT runs on raw step-axis losses (segment cost is identical under
  // X transforms); we then translate the resulting changepoint X values
  // through the mapper for the markers.
  const changepoints = useMemo(
    () => buildChangepoints(metrics.data),
    [metrics.data],
  )

  const allMarkers = useMemo(
    () =>
      buildAllMarkers({
        metrics: metrics.data,
        xMode,
        checkpointMarkers,
        changepointSteps: changepoints.changepointSteps,
      }),
    [checkpointMarkers, changepoints.changepointSteps, metrics.data, xMode],
  )

  const totalPoints = metrics.data?.loss?.length ?? 0
  const epochAvailable = (metrics.data?.loss ?? []).some(
    (point) => typeof point.epoch === "number",
  )
  useEffect(() => {
    if (xMode !== "epoch" || metrics.data == null || epochAvailable) return
    selectXMode("step")
  }, [epochAvailable, metrics.data, xMode])
  // Estimated training progress in [0..1] for context-aware tone
  // selection in the effectiveness panel. Prefers the trainer-reported
  // total_steps over the config-derived fallback.
  const progress: number | null = useMemo(() => {
    const lossArr = metrics.data?.loss ?? []
    const lastStep = lossArr.length
      ? lossArr[lossArr.length - 1].step
      : typeof metrics.data?.last_step === "number"
        ? metrics.data.last_step
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
  // Top-level view: 内置训练分析 vs W&B 数据
  const [viewKind, setViewKind] = useState<ViewKind>(() => {
    if (typeof window === "undefined") return "builtin"
    const stored = window.localStorage.getItem("lorahub.analysis.view-kind")
    return stored === "wandb" ? "wandb" : "builtin"
  })
  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem("lorahub.analysis.view-kind", viewKind)
  }, [viewKind])
  const samplesCount = files.data?.samples?.length ?? 0
  const tableRowCount = totalPoints
  const seriesCount = lossSeries.length
  const aiState: "ready" | "missing" = aiCache.data?.analysis ? "ready" : "missing"

  // wandb state — read from the same JobDetail that powers the rest
  // of the workbench. ``enable_wandb`` lives at the top-level
  // [monitoring] section per ``MonitoringConfig``; the run URL comes
  // from JobRecord.metadata once `wandb.init()` prints its banner.
  const wandbEnabled = useMemo(() => {
    const cfg = jobDetail?.config_snapshot as Record<string, unknown> | undefined
    const m = cfg?.["monitoring"] as Record<string, unknown> | undefined
    const bag = (cfg?.["backend"] as Record<string, unknown> | undefined)?.[
      "diffusionPipe"
    ] as Record<string, unknown> | undefined
    // Top-level wins; legacy DiffusionPipeOptions block is the fallback.
    return Boolean(m?.["enableWandb"]) || Boolean(bag?.["enableWandb"])
  }, [jobDetail])
  const wandbRunUrl = useMemo(() => {
    const meta = jobDetail?.metadata as Record<string, unknown> | undefined
    const url = meta?.["wandb_run_url"]
    return typeof url === "string" && url ? url : null
  }, [jobDetail])

  // sampling.triggerWord 是模板替换后的实际触发词；为 LoRA 预览角标提供文字。
  // 后端在 lifecycle._resolve_trigger_word 里要么用配置里的值，要么从数据集
  // 推断；这里只读快照里的最终值，未设时回退到 base name。
  const samplingTriggerWord = useMemo<string | null>(() => {
    const cfg = jobDetail?.config_snapshot as Record<string, unknown> | undefined
    const sampling = cfg?.["sampling"] as Record<string, unknown> | undefined
    const raw = sampling?.["triggerWord"] ?? sampling?.["trigger_word"]
    return typeof raw === "string" && raw.trim() ? raw.trim() : null
  }, [jobDetail])
  const analysisMotionRef = useAnimeAnalysisMotion<HTMLDivElement>([
    job.id,
    viewKind,
    viewMode,
    xMode,
    panels.showStageTimeline,
    panels.showMetricGrid,
    panels.showCheckpointPlayback,
  ])

  return (
    <div className="flex flex-col min-h-0">
      <div className="overflow-x-auto border-b border-border/40 bg-background/40 px-4 pt-3 md:px-7 md:pt-4">
        <Tabs value={viewKind} onValueChange={(v) => setViewKind(v as ViewKind)}>
          <TabsList variant="line" className="gap-3">
            <TabsTrigger value="builtin" className="text-[12px]">
              训练分析
            </TabsTrigger>
            <TabsTrigger value="wandb" className="text-[12px]">
              W&amp;B
              {wandbRunUrl ? (
                <span className="ml-1 text-emerald-600 dark:text-emerald-400">✓</span>
              ) : wandbEnabled ? (
                <span className="ml-1 text-muted-foreground/80">等待</span>
              ) : (
                <span className="ml-1 text-muted-foreground/80">未启用</span>
              )}
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {viewKind === "wandb" ? (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <WandbTab jobId={job.id} enabled={wandbEnabled} runUrl={wandbRunUrl} />
        </div>
      ) : (
        <>
          <AnalysisKpiStrip
            job={job}
            fallbackTotalSteps={fallbackTotalSteps}
            backend={backendInfo}
          />

      <div ref={analysisMotionRef} className="space-y-4 px-4 py-4 md:px-7">
        <BackendContextStrip
          backend={backendInfo}
          metrics={metrics.data ?? null}
        />
        {crossBackendReference && (
          <div className="border-l-2 border-amber-500/70 bg-amber-500/8 px-3 py-2 text-[11px] text-muted-foreground">
            当前基线来自 {referenceBackendInfo.label}。不同后端的损失定义不可直接叠加；此处保留当前任务曲线，请在多任务对比中查看归一化形态。
          </div>
        )}
        {/* View-mode switcher: live / postmortem / custom. Mode picks
            sensible defaults for which heavy panels are open by
            default; manual toggles flip into custom and persist. */}
        <ViewModeSwitcher
          mode={viewMode}
          panels={panels}
          isTerminal={isTerminal}
          xMode={xMode}
          epochAvailable={epochAvailable}
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
          backend={backendInfo}
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
                      totalPoints > MAX_POINTS
                        ? `（保真下采样到 ${MAX_POINTS}）`
                        : ""
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
            <MetricGrid
              metrics={metrics.data ?? null}
              backend={backendInfo}
              jobId={job.id}
              xMode={xMode}
            />
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
            triggerWord={samplingTriggerWord}
          />
        )}

        {/* Bottom tabs — stats / table / samples / AI. */}
        <Tabs
          value={bottomTab}
          onValueChange={(v) => setBottomTab(v as BottomTabKey)}
        >
          <div className="overflow-x-auto rounded-t-[6px] border-y border-border/60 bg-muted/40 px-3.5 py-1.5">
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
              triggerWord={samplingTriggerWord}
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
        </>
      )}
    </div>
  )
}
