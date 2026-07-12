import type { BackendId, JobMetricsResponse } from "@/lib/api"
import type {
  ChartBand,
  ChartMarker,
  LossSeries,
} from "../../jobs/components/loss-chart-model"
import {
  defaultRollingWindow,
  rollingQuartiles,
  type BandPoint,
} from "./loss-stats"
import { analyseChangepoints } from "./pelt"
import type { ReferenceRun } from "./reference-run"
import { xMapper, type XMode } from "./x-axis-mode"

const EMA_ALPHA = 0.1

export function buildLossSeries({
  metrics,
  jobId,
  xMode,
  referenceRun,
  referenceMetrics,
  backendType,
  referenceBackendType,
}: {
  metrics: JobMetricsResponse | null | undefined
  jobId: string
  xMode: XMode
  referenceRun: ReferenceRun | null
  referenceMetrics: JobMetricsResponse | null | undefined
  backendType: BackendId | null
  referenceBackendType: BackendId | null
}): LossSeries[] {
  const allLossPoints = (metrics?.loss ?? []).filter(
    (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
      typeof p.loss === "number" && Number.isFinite(p.loss),
  )
  const map = xMapper(xMode, metrics ?? null)
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
    out.push({
      id: `${jobId}-train-raw`,
      label: "原始 loss · 每步",
      color: "var(--chart-1)",
      points: trainPoints,
    })
    const rollingWindow = defaultRollingWindow(trainPoints.length)
    const robust =
      trainPoints.length >= 8
        ? rollingQuartiles(trainPoints, rollingWindow)
        : null
    if (robust) {
      out.push({
        id: `${jobId}-train-median`,
        label: `滚动中位数 · ${rollingWindow} 步`,
        color: "var(--chart-2)",
        dashed: true,
        points: robust.median,
      })
    }
    if (trainPoints.length >= 6) {
      let acc = trainPoints[0].loss
      const ema = trainPoints.map((p) => {
        acc = EMA_ALPHA * p.loss + (1 - EMA_ALPHA) * acc
        return { step: p.step, loss: acc }
      })
      out.push({
        id: `${jobId}-train-ema`,
        label: `EMA α=${EMA_ALPHA}`,
        color: "var(--chart-3)",
        dashed: true,
        points: ema,
      })
    }
  }

  const valPoints = (metrics?.val_loss ?? []).filter(
    (p): p is { epoch: number; val_loss: number; step?: number | null; ts: number } =>
      typeof p.val_loss === "number" && Number.isFinite(p.val_loss),
  )
  if (valPoints.length > 0) {
    const epochToStep = new Map<number, number>()
    for (const p of metrics?.loss ?? []) {
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
      id: `${jobId}-val`,
      label: "验证 loss",
      color: "var(--chart-5)",
      points: mapped,
    })
  }

  const comparableReference =
    backendType != null &&
    referenceBackendType != null &&
    backendType === referenceBackendType
  if (
    referenceRun &&
    referenceRun.jobId !== jobId &&
    referenceMetrics &&
    comparableReference
  ) {
    const refMap = xMapper(xMode, referenceMetrics)
    const refPoints = (referenceMetrics.loss ?? [])
      .filter(
        (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
          typeof p.loss === "number" && Number.isFinite(p.loss),
      )
      .map((p) => ({ step: refMap(p), loss: p.loss }))
    if (refPoints.length > 0) {
      out.push({
        id: `${jobId}-ref`,
        label: `参考 ${referenceRun.label}`,
        color: "color-mix(in oklch, var(--muted-foreground) 80%, transparent)",
        dashed: true,
        points: refPoints,
      })
    }
    if (comparableReference) {
      const refEpochToStep = new Map<number, number>()
      for (const point of referenceMetrics.loss ?? []) {
        if (typeof point.epoch === "number" && typeof point.step === "number") {
          refEpochToStep.set(point.epoch, point.step)
        }
      }
      const refLastStep = referenceMetrics.last_step ?? 0
      const refVal = (referenceMetrics.val_loss ?? [])
        .filter(
          (point): point is {
            epoch: number
            val_loss: number
            step?: number | null
            ts: number
          } =>
            typeof point.val_loss === "number" &&
            Number.isFinite(point.val_loss),
        )
        .map((point) => ({
          step: refMap({
            step:
              typeof point.step === "number"
                ? point.step
                : (refEpochToStep.get(point.epoch) ??
                  (refLastStep || point.epoch)),
            epoch: point.epoch,
            ts: point.ts,
          }),
          loss: point.val_loss,
        }))
      if (refVal.length > 0) {
        out.push({
          id: `${jobId}-ref-val`,
          label: `参考 ${referenceRun.label} · 验证`,
          color: "var(--chart-4)",
          dashed: true,
          points: refVal,
        })
      }
    }
  }
  return out
}

export function buildCheckpointMarkers(
  metrics: JobMetricsResponse | null | undefined,
  xMode: XMode,
): ChartMarker[] {
  const map = xMapper(xMode, metrics ?? null)
  return (metrics?.checkpoints ?? [])
    .filter((c): c is { path: string; step: number; ts: number } =>
      typeof c.step === "number" && Number.isFinite(c.step),
    )
    .map((c) => ({
      step: map(c),
      label: c.path,
      color: "var(--chart-3)",
    }))
}

export function buildLossBands(
  metrics: JobMetricsResponse | null | undefined,
  xMode: XMode,
): ChartBand[] {
  const points = (metrics?.loss ?? []).filter(
    (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
      typeof p.loss === "number" && Number.isFinite(p.loss),
  )
  if (points.length < 8) return []
  const map = xMapper(xMode, metrics ?? null)
  const series: BandPoint[] = rollingQuartiles(
    points.map((p) => ({ step: map(p), loss: p.loss })),
  ).band
  return [
    {
      id: "train-iqr",
      label: "局部 loss 分布 · Q25–Q75",
      color: "color-mix(in oklch, var(--chart-1) 14%, transparent)",
      points: series,
    },
  ]
}

export function buildChangepoints(
  metrics: JobMetricsResponse | null | undefined,
) {
  const points = (metrics?.loss ?? [])
    .filter(
      (p): p is { step: number; loss: number; ts: number } =>
        typeof p.loss === "number" && Number.isFinite(p.loss),
    )
    .map((p) => ({ step: p.step, loss: p.loss }))
  return analyseChangepoints(points)
}

export function buildAllMarkers({
  metrics,
  xMode,
  checkpointMarkers,
  changepointSteps,
}: {
  metrics: JobMetricsResponse | null | undefined
  xMode: XMode
  checkpointMarkers: ChartMarker[]
  changepointSteps: number[]
}): ChartMarker[] {
  const map = xMapper(xMode, metrics ?? null)
  const stepToSample = new Map<
    number,
    { step: number; epoch?: number | null; ts: number }
  >()
  for (const p of metrics?.loss ?? []) {
    if (typeof p.step === "number")
      stepToSample.set(p.step, { step: p.step, epoch: p.epoch, ts: p.ts })
  }
  const out: ChartMarker[] = [...checkpointMarkers]
  for (const s of changepointSteps) {
    const sample = stepToSample.get(s) ?? { step: s, ts: 0 }
    out.push({ step: map(sample), color: "var(--chart-2)" })
  }
  for (const point of metrics?.nonfinite_loss ?? []) {
    if (typeof point.step !== "number") continue
    out.push({
      step: map({ step: point.step, ts: point.ts ?? undefined }),
      label: "训练 loss 为 NaN/Inf",
      color: "var(--destructive)",
    })
  }
  for (const point of metrics?.nonfinite_val_loss ?? []) {
    if (typeof point.step !== "number") continue
    out.push({
      step: map({ step: point.step, ts: point.ts ?? undefined }),
      label: "验证 loss 为 NaN/Inf",
      color: "var(--destructive)",
    })
  }
  return out
}
