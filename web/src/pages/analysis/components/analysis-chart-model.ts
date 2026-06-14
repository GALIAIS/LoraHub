import type { JobMetricsResponse } from "@/lib/api"
import type {
  ChartBand,
  ChartMarker,
  LossSeries,
} from "../../jobs/components/loss-chart"
import { rollingQuartiles, type BandPoint } from "./loss-stats"
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
}: {
  metrics: JobMetricsResponse | null | undefined
  jobId: string
  xMode: XMode
  referenceRun: ReferenceRun | null
  referenceMetrics: JobMetricsResponse | null | undefined
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
    const robust =
      trainPoints.length >= 8 ? rollingQuartiles(trainPoints) : null
    if (robust) {
      out.push({
        id: `${jobId}-train-median`,
        label: "训练 loss · 中位数",
        color: "var(--chart-1)",
        points: robust.median,
      })
      out.push({
        id: `${jobId}-train-raw`,
        label: "原始采样",
        color: "var(--chart-1)",
        dashed: true,
        points: trainPoints,
      })
    } else {
      out.push({
        id: `${jobId}-train`,
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
        id: `${jobId}-train-ema`,
        label: `EMA α=${EMA_ALPHA}`,
        color: "var(--chart-1)",
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
      color: "var(--chart-2)",
      points: mapped,
    })
  }

  if (referenceRun && referenceRun.jobId !== jobId && referenceMetrics) {
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
      color: "color-mix(in oklch, var(--chart-1) 18%, transparent)",
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
  return out
}
