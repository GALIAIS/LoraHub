/**
 * MetricsTable — sortable + CSV-exportable per-step training metrics.
 * The validation column is dropped entirely when no `val_loss` events
 * exist (instead of a wall of "—" placeholders).
 */
import { useMemo, useState } from "react"
import { Download } from "lucide-react"
import type { JobMetricPoint, JobValLossPoint } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

interface MetricRow {
  step: number
  epoch: number | null
  trainLoss: number | null
  valLoss: number | null
  ts: number
}

type SortKey = "step" | "epoch" | "trainLoss" | "valLoss" | "ts"
type SortDir = "asc" | "desc"

export function MetricsTable({
  loss,
  valLoss,
  loading,
}: {
  loss: JobMetricPoint[]
  valLoss: JobValLossPoint[]
  loading: boolean
}) {
  const [sortKey, setSortKey] = useState<SortKey>("step")
  const [sortDir, setSortDir] = useState<SortDir>("desc")

  const hasValLoss = valLoss.some(
    (v) => typeof v.val_loss === "number" && Number.isFinite(v.val_loss),
  )

  const rows = useMemo<MetricRow[]>(() => {
    const valByEpoch = new Map<number, number>()
    for (const v of valLoss) {
      if (typeof v.val_loss === "number" && Number.isFinite(v.val_loss)) {
        valByEpoch.set(v.epoch, v.val_loss)
      }
    }
    return loss
      .filter((p) => typeof p.loss === "number" && Number.isFinite(p.loss))
      .map((p) => ({
        step: p.step,
        epoch: typeof p.epoch === "number" ? p.epoch : null,
        trainLoss: typeof p.loss === "number" ? p.loss : null,
        valLoss:
          typeof p.epoch === "number" && valByEpoch.has(p.epoch)
            ? (valByEpoch.get(p.epoch) ?? null)
            : null,
        ts: p.ts,
      }))
  }, [loss, valLoss])

  const sortedRows = useMemo(() => {
    const out = [...rows]
    out.sort((a, b) => {
      const va = a[sortKey]
      const vb = b[sortKey]
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      const cmp = va < vb ? -1 : va > vb ? 1 : 0
      return sortDir === "asc" ? cmp : -cmp
    })
    return out
  }, [rows, sortKey, sortDir])

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc")
    } else {
      setSortKey(key)
      setSortDir("desc")
    }
  }

  function exportCsv() {
    const baseHeader = ["step", "epoch", "train_loss", "timestamp"]
    const header = hasValLoss
      ? [...baseHeader.slice(0, 3), "val_loss", "timestamp"]
      : baseHeader
    const lines = [header.join(",")]
    for (const r of sortedRows) {
      const cols: Array<string | number> = [
        r.step,
        r.epoch ?? "",
        r.trainLoss ?? "",
      ]
      if (hasValLoss) cols.push(r.valLoss ?? "")
      cols.push(new Date(r.ts * 1000).toISOString())
      lines.push(cols.join(","))
    }
    const blob = new Blob([lines.join("\n")], {
      type: "text/csv;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "metrics.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card className="rounded-[6px] border-border/60">
      <CardHeader className="py-2.5 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          指标表格
        </CardTitle>
        <div className="flex items-center gap-2">
          {!hasValLoss && (
            <span
              className="text-[10px] text-muted-foreground/70"
              title="recipe 未配置 validation.valSplit，所以验证 loss 列已隐藏"
            >
              未启用验证集
            </span>
          )}
          <span className="text-[10px] text-muted-foreground/70">
            {loading ? "加载中…" : `共 ${rows.length} 行`}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={rows.length === 0}
            onClick={exportCsv}
            className="h-7 text-[11px]"
          >
            <Download className="size-3" /> 导出 CSV
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[360px]">
          <table className="w-full text-[11px] tabular-nums">
            <thead className="sticky top-0 bg-background/95 backdrop-blur border-b border-border/60">
              <tr className="text-left text-muted-foreground">
                <SortHeader k="step" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  step
                </SortHeader>
                <SortHeader k="epoch" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  epoch
                </SortHeader>
                <SortHeader k="trainLoss" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  train loss
                </SortHeader>
                {hasValLoss && (
                  <SortHeader k="valLoss" current={sortKey} dir={sortDir} onClick={toggleSort}>
                    val loss
                  </SortHeader>
                )}
                <SortHeader k="ts" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  时间
                </SortHeader>
              </tr>
            </thead>
            <tbody>
              {sortedRows.length === 0 && !loading && (
                <tr>
                  <td
                    colSpan={hasValLoss ? 5 : 4}
                    className="py-8 text-center text-muted-foreground"
                  >
                    暂无指标数据
                  </td>
                </tr>
              )}
              {sortedRows.map((r) => (
                <tr
                  key={`${r.step}-${r.ts}`}
                  className="border-b border-border/30 hover:bg-muted/30"
                >
                  <td className="px-3 py-1.5">{r.step}</td>
                  <td className="px-3 py-1.5">{r.epoch ?? "—"}</td>
                  <td className="px-3 py-1.5">
                    {r.trainLoss != null ? r.trainLoss.toFixed(4) : "—"}
                  </td>
                  {hasValLoss && (
                    <td className="px-3 py-1.5">
                      {r.valLoss != null ? r.valLoss.toFixed(4) : "—"}
                    </td>
                  )}
                  <td className="px-3 py-1.5 text-muted-foreground">
                    {new Date(r.ts * 1000).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function SortHeader({
  k,
  current,
  dir,
  onClick,
  children,
}: {
  k: SortKey
  current: SortKey
  dir: SortDir
  onClick: (k: SortKey) => void
  children: React.ReactNode
}) {
  const active = k === current
  return (
    <th
      className="px-3 py-2 font-medium text-[10px] uppercase tracking-[0.12em] cursor-pointer select-none"
      onClick={() => onClick(k)}
    >
      <span className="inline-flex items-center gap-1">
        {children}
        {active && (
          <span className="text-foreground/70">{dir === "asc" ? "↑" : "↓"}</span>
        )}
      </span>
    </th>
  )
}
