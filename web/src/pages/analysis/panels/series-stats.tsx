/**
 * SeriesStatsCard — count / latest / mean / std / peak / trough table
 * across every chartable series in the active job. Surfaces the same
 * numbers as the chart legends but in tabular form.
 */
import { useMemo } from "react"
import type { JobMetricsResponse } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { LossSeries } from "../../jobs/components/loss-chart"

interface StatRow {
  id: string
  label: string
  color: string
  count: number
  min: number
  max: number
  avg: number
  std: number
  latest: number
}

function describeSeries(
  id: string,
  label: string,
  color: string,
  values: number[],
): StatRow | null {
  const filtered = values.filter((v) => Number.isFinite(v))
  if (filtered.length === 0) return null
  const min = Math.min(...filtered)
  const max = Math.max(...filtered)
  const avg = filtered.reduce((a, b) => a + b, 0) / filtered.length
  const variance =
    filtered.reduce((a, b) => a + (b - avg) ** 2, 0) / filtered.length
  const std = Math.sqrt(variance)
  return {
    id,
    label,
    color,
    count: filtered.length,
    min,
    max,
    avg,
    std,
    latest: filtered[filtered.length - 1],
  }
}

function fmtVal(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  if (Math.abs(v) >= 0.01) return v.toFixed(4)
  if (v === 0) return "0"
  return v.toExponential(2)
}

export function SeriesStatsCard({
  series,
  metrics,
}: {
  series: LossSeries[]
  metrics: JobMetricsResponse | null
}) {
  const rows: StatRow[] = useMemo(() => {
    const out: StatRow[] = []
    for (const s of series) {
      const r = describeSeries(
        s.id,
        s.label,
        s.color,
        s.points.map((p) => p.loss),
      )
      if (r) out.push(r)
    }
    const lr =
      metrics?.loss
        ?.filter((p) => typeof p.lr === "number" && Number.isFinite(p.lr))
        .map((p) => p.lr as number) ?? []
    const r1 = describeSeries("lr", "学习率", "var(--chart-1)", lr)
    if (r1) out.push(r1)
    const it =
      metrics?.loss
        ?.filter(
          (p) =>
            typeof p.iter_time_s === "number" &&
            Number.isFinite(p.iter_time_s),
        )
        .map((p) => p.iter_time_s as number) ?? []
    const r2 = describeSeries("it", "迭代时长 s", "var(--chart-3)", it)
    if (r2) out.push(r2)
    const sps =
      metrics?.loss
        ?.filter(
          (p) =>
            typeof p.samples_per_sec === "number" &&
            Number.isFinite(p.samples_per_sec),
        )
        .map((p) => p.samples_per_sec as number) ?? []
    const r3 = describeSeries("sps", "样本/秒", "var(--chart-4)", sps)
    if (r3) out.push(r3)
    return out
  }, [series, metrics])

  return (
    <Card>
      <CardHeader className="py-2.5 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          序列统计
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {rows.length === 0 ? "无数据" : `共 ${rows.length} 条曲线`}
        </span>
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <div className="px-4 py-6 text-xs text-muted-foreground">
            尚无可统计的曲线。
          </div>
        ) : (
          <div className="overflow-x-auto">
          <table className="min-w-[620px] w-full text-[11px] tabular-nums">
            <thead className="border-b border-border/60 text-muted-foreground">
              <tr className="text-left">
                <th className="px-3 py-2 font-medium text-[10px] uppercase tracking-[0.12em]">
                  曲线
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  采样数
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  最新
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  均值
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  标准差
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  峰值
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  谷值
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.id}
                  className="border-b border-border/30 last:border-b-0"
                >
                  <td className="px-3 py-1.5">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="inline-block size-2 rounded-full"
                        style={{ background: r.color }}
                        aria-hidden
                      />
                      <span className="text-foreground/85">{r.label}</span>
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {r.count}
                  </td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.latest)}</td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.avg)}</td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {fmtVal(r.std)}
                  </td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.max)}</td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.min)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
