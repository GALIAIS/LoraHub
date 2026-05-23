/**
 * Metric grid — TensorBoard-style breakdown of every chartable scalar.
 *
 * Each metric lives in its own small card with:
 *   - A tiny path-style title (`loss/average`, `loss/ema`, ...).
 *   - One `<MultiLineChart>` (so the user gets the full
 *     wheel-zoom / pan / fullscreen / CSV stack we already built).
 *   - A single inline summary value (last point) on the header.
 *
 * Cards arranged in a responsive grid (1 / 2 / 3 columns). Cards whose
 * underlying series are empty are *omitted* — no "no data" placeholders
 * cluttering the page. The user can still see them once data starts
 * flowing.
 *
 * Hardware-side metrics (gpu util / vram / temperature) live on the
 * dashboard's GPU page and the job-detail realtime tile; this grid is
 * about *training effectiveness*, not host-load monitoring.
 */
import { useMemo } from "react"
import type { JobMetricsResponse } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MultiLineChart, type MultiLineSeries } from "./multi-line-chart"

const EMA_ALPHA = 0.1

interface MetricCardSpec {
  id: string
  title: string
  series: MultiLineSeries[]
  /** Inline last-value chip rendered to the right of the title. */
  latest?: string
  xLabel?: string
}

export function MetricGrid({
  metrics,
  jobId,
}: {
  metrics: JobMetricsResponse | null
  jobId: string
}) {
  const cards = useMemo<MetricCardSpec[]>(() => {
    const out: MetricCardSpec[] = []
    const losses =
      metrics?.loss?.filter(
        (p) => typeof p.loss === "number" && Number.isFinite(p.loss),
      ) ?? []

    // ---------- loss/raw ----------
    if (losses.length > 0) {
      out.push({
        id: "loss-raw",
        title: "loss/raw",
        latest: fmtFloat(losses[losses.length - 1].loss as number, 4),
        xLabel: "step",
        series: [
          {
            id: "raw",
            label: "训练 loss",
            color: "var(--chart-1)",
            points: losses.map((p) => ({
              x: p.step,
              y: p.loss as number,
            })),
          },
        ],
      })
    }

    // ---------- loss/ema (EMA-smoothed train loss) ----------
    if (losses.length >= 6) {
      let acc = losses[0].loss as number
      const ema = losses.map((p) => {
        acc = EMA_ALPHA * (p.loss as number) + (1 - EMA_ALPHA) * acc
        return { x: p.step, y: acc }
      })
      out.push({
        id: "loss-ema",
        title: `loss/ema · α=${EMA_ALPHA}`,
        latest: fmtFloat(ema[ema.length - 1].y, 4),
        xLabel: "step",
        series: [
          {
            id: "ema",
            label: "EMA",
            color: "var(--chart-2)",
            points: ema,
          },
        ],
      })
    }

    // ---------- loss/epoch_avg (mean train loss per epoch) ----------
    if (metrics?.loss && metrics.loss.length > 0) {
      const buckets = new Map<number, { sum: number; n: number; lastStep: number }>()
      for (const p of metrics.loss) {
        if (typeof p.loss !== "number" || !Number.isFinite(p.loss)) continue
        if (typeof p.epoch !== "number") continue
        const cur = buckets.get(p.epoch) ?? { sum: 0, n: 0, lastStep: p.step }
        cur.sum += p.loss
        cur.n += 1
        cur.lastStep = p.step
        buckets.set(p.epoch, cur)
      }
      const points = Array.from(buckets.entries())
        .map(([epoch, v]) => ({ x: epoch, y: v.sum / v.n, lastStep: v.lastStep }))
        .sort((a, b) => a.x - b.x)
      if (points.length >= 2) {
        out.push({
          id: "loss-epoch",
          title: "loss/epoch_avg",
          latest: fmtFloat(points[points.length - 1].y, 4),
          xLabel: "epoch",
          series: [
            {
              id: "epoch_avg",
              label: "epoch 均值",
              color: "var(--chart-3)",
              points: points.map((p) => ({ x: p.x, y: p.y })),
            },
          ],
        })
      }
    }

    // ---------- val_loss (if any) ----------
    const vals =
      metrics?.val_loss?.filter(
        (p) => typeof p.val_loss === "number" && Number.isFinite(p.val_loss),
      ) ?? []
    if (vals.length > 0) {
      out.push({
        id: "loss-val",
        title: "loss/val",
        latest: fmtFloat(vals[vals.length - 1].val_loss, 4),
        xLabel: "epoch",
        series: [
          {
            id: "val",
            label: "验证 loss",
            color: "var(--chart-4)",
            points: vals.map((p) => ({ x: p.epoch, y: p.val_loss })),
          },
        ],
      })
    }

    // ---------- lr ----------
    const lr =
      metrics?.loss?.filter(
        (p) => typeof p.lr === "number" && Number.isFinite(p.lr),
      ) ?? []
    if (lr.length > 0) {
      const last = lr[lr.length - 1].lr as number
      out.push({
        id: "lr",
        title: "schedule/learning_rate",
        latest: fmtSci(last),
        xLabel: "step",
        series: [
          {
            id: "lr",
            label: "学习率",
            color: "var(--chart-1)",
            points: lr.map((p) => ({ x: p.step, y: p.lr as number })),
          },
        ],
      })
    }

    // ---------- iter_time_s ----------
    const it =
      metrics?.loss?.filter(
        (p) =>
          typeof p.iter_time_s === "number" && Number.isFinite(p.iter_time_s),
      ) ?? []
    if (it.length > 0) {
      out.push({
        id: "iter-time",
        title: "throughput/iter_time_s",
        latest: `${fmtFloat(it[it.length - 1].iter_time_s as number, 2)} s`,
        xLabel: "step",
        series: [
          {
            id: "iter",
            label: "迭代时长",
            color: "var(--chart-3)",
            unit: "s",
            points: it.map((p) => ({
              x: p.step,
              y: p.iter_time_s as number,
            })),
          },
        ],
      })
    }

    // ---------- samples_per_sec ----------
    const sps =
      metrics?.loss?.filter(
        (p) =>
          typeof p.samples_per_sec === "number" &&
          Number.isFinite(p.samples_per_sec),
      ) ?? []
    if (sps.length > 0) {
      out.push({
        id: "sps",
        title: "throughput/samples_per_sec",
        latest: fmtFloat(sps[sps.length - 1].samples_per_sec as number, 2),
        xLabel: "step",
        series: [
          {
            id: "sps",
            label: "样本/秒",
            color: "var(--chart-4)",
            points: sps.map((p) => ({
              x: p.step,
              y: p.samples_per_sec as number,
            })),
          },
        ],
      })
    }

    // ---------- lora/effective_rank, top1_energy, fro_norm ----------
    const spectrum = metrics?.lora_spectrum ?? []
    if (spectrum.length > 0) {
      const eff = spectrum
        .filter(
          (s) =>
            typeof s.effective_rank === "number" &&
            Number.isFinite(s.effective_rank) &&
            typeof s.step === "number",
        )
        .map((s) => ({ x: s.step as number, y: s.effective_rank as number }))
      if (eff.length > 0) {
        out.push({
          id: "lora-eff-rank",
          title: "lora/effective_rank",
          latest: fmtFloat(eff[eff.length - 1].y, 2),
          xLabel: "step",
          series: [
            {
              id: "eff",
              label: "有效秩",
              color: "var(--chart-1)",
              points: eff,
            },
          ],
        })
      }
      const top1 = spectrum
        .filter(
          (s) =>
            typeof s.top1_energy === "number" &&
            Number.isFinite(s.top1_energy) &&
            typeof s.step === "number",
        )
        .map((s) => ({ x: s.step as number, y: (s.top1_energy as number) * 100 }))
      if (top1.length > 0) {
        out.push({
          id: "lora-top1",
          title: "lora/top1_energy",
          latest: `${fmtFloat(top1[top1.length - 1].y, 1)} %`,
          xLabel: "step",
          series: [
            {
              id: "top1",
              label: "首奇异值能量占比",
              color: "var(--chart-2)",
              unit: "%",
              points: top1,
            },
          ],
        })
      }
      const fro = spectrum
        .filter(
          (s) =>
            typeof s.fro_norm === "number" &&
            Number.isFinite(s.fro_norm) &&
            typeof s.step === "number",
        )
        .map((s) => ({ x: s.step as number, y: s.fro_norm as number }))
      if (fro.length > 0) {
        out.push({
          id: "lora-fro",
          title: "lora/fro_norm",
          latest: fmtFloat(fro[fro.length - 1].y, 3),
          xLabel: "step",
          series: [
            {
              id: "fro",
              label: "ΔW 弗罗贝尼乌斯范数",
              color: "var(--chart-3)",
              points: fro,
            },
          ],
        })
      }
    }

    return out
  }, [metrics])

  if (cards.length === 0) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-xs text-muted-foreground">
          暂无可绘制的指标。训练后端产出第一行
          <code className="font-mono mx-1">steps: N loss: …</code>
          后会自动出现。
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      {cards.map((c) => (
        <Card key={c.id}>
          <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
            <CardTitle className="text-[10.5px] tracking-[0.16em] text-foreground/85 font-mono">
              {c.title}
            </CardTitle>
            {c.latest != null && (
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {c.latest}
              </span>
            )}
          </CardHeader>
          <CardContent className="p-3">
            <MultiLineChart
              series={c.series}
              xLabel={c.xLabel}
              persistKey={`${jobId}.${c.id}`}
              title={c.title}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function fmtFloat(v: number, digits: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  return v.toFixed(digits)
}

function fmtSci(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (v === 0) return "0"
  const abs = Math.abs(v)
  if (abs >= 1e-3 && abs < 1e3) return v.toFixed(4)
  return v.toExponential(2)
}
