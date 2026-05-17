import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Download, ImageIcon, X } from "lucide-react"
import { api } from "@/lib/api"
import type { JobMetricPoint, JobFile, JobValLossPoint } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { TERMINAL_STATES } from "../utils"

interface AnalysisTabProps {
  jobId: string
  jobState: string | undefined
}

interface MetricRow {
  step: number
  epoch: number | null
  trainLoss: number | null
  valLoss: number | null
  ts: number
}

type SortKey = "step" | "epoch" | "trainLoss" | "valLoss" | "ts"
type SortDir = "asc" | "desc"

export function AnalysisTab({ jobId, jobState }: AnalysisTabProps) {
  const isTerminal = jobState ? TERMINAL_STATES.has(jobState) : false

  const metrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: isTerminal ? false : 4000,
  })
  const files = useQuery({
    queryKey: ["job-files", jobId],
    queryFn: () => api.getJobFiles(jobId),
    refetchInterval: isTerminal ? false : 8000,
  })

  return (
    <div className="space-y-4">
      <MetricsTable
        loss={metrics.data?.loss ?? []}
        valLoss={metrics.data?.val_loss ?? []}
        loading={metrics.isLoading}
      />
      <SamplesGallery
        jobId={jobId}
        samples={files.data?.samples ?? []}
        loading={files.isLoading}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Metrics table
// ---------------------------------------------------------------------------

function MetricsTable({
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

  const rows = useMemo<MetricRow[]>(() => {
    // Index validation loss by epoch so we can hang it off the
    // matching train rows. Train rows that aren't on an eval boundary
    // simply leave valLoss=null, which renders blank.
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
    const header = ["step", "epoch", "train_loss", "val_loss", "timestamp"]
    const lines = [header.join(",")]
    for (const r of sortedRows) {
      lines.push(
        [
          r.step,
          r.epoch ?? "",
          r.trainLoss ?? "",
          r.valLoss ?? "",
          new Date(r.ts * 1000).toISOString(),
        ].join(","),
      )
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "metrics.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          指标表格
        </CardTitle>
        <div className="flex items-center gap-2">
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
                <SortHeader k="valLoss" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  val loss
                </SortHeader>
                <SortHeader k="ts" current={sortKey} dir={sortDir} onClick={toggleSort}>
                  时间
                </SortHeader>
              </tr>
            </thead>
            <tbody>
              {sortedRows.length === 0 && !loading && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-muted-foreground">
                    暂无指标数据
                  </td>
                </tr>
              )}
              {sortedRows.map((r) => (
                <tr key={`${r.step}-${r.ts}`} className="border-b border-border/30 hover:bg-muted/30">
                  <td className="px-3 py-1.5">{r.step}</td>
                  <td className="px-3 py-1.5">{r.epoch ?? "—"}</td>
                  <td className="px-3 py-1.5">
                    {r.trainLoss != null ? r.trainLoss.toFixed(4) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.valLoss != null ? r.valLoss.toFixed(4) : "—"}
                  </td>
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
        {active && <span className="text-foreground/70">{dir === "asc" ? "↑" : "↓"}</span>}
      </span>
    </th>
  )
}

// ---------------------------------------------------------------------------
// Samples gallery
// ---------------------------------------------------------------------------

interface SampleEntry extends JobFile {
  // Extracted from the filename (e.g. "epoch5-step120.png" -> 5)
  epoch: number | null
  step: number | null
}

function parseSampleMeta(path: string): { epoch: number | null; step: number | null } {
  const name = path.split(/[\\/]/).pop() ?? ""
  const epochMatch = name.match(/e(?:poch)?[-_]?(\d+)/i)
  const stepMatch = name.match(/s(?:tep)?[-_]?(\d+)/i)
  return {
    epoch: epochMatch ? Number(epochMatch[1]) : null,
    step: stepMatch ? Number(stepMatch[1]) : null,
  }
}

function SamplesGallery({
  jobId,
  samples,
  loading,
}: {
  jobId: string
  samples: JobFile[]
  loading: boolean
}) {
  const [openSrc, setOpenSrc] = useState<string | null>(null)

  const enriched = useMemo<SampleEntry[]>(() => {
    return samples
      .map((s) => ({ ...s, ...parseSampleMeta(s.path) }))
      .sort((a, b) => {
        // Group by epoch ascending, then step ascending, then mtime.
        const ae = a.epoch ?? Number.MAX_SAFE_INTEGER
        const be = b.epoch ?? Number.MAX_SAFE_INTEGER
        if (ae !== be) return ae - be
        const as = a.step ?? Number.MAX_SAFE_INTEGER
        const bs = b.step ?? Number.MAX_SAFE_INTEGER
        if (as !== bs) return as - bs
        return a.modified_at - b.modified_at
      })
  }, [samples])

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          样本预览画廊
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading ? "加载中…" : `${enriched.length} 张样本`}
        </span>
      </CardHeader>
      <CardContent className="p-4">
        {enriched.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground">
            <ImageIcon className="size-8 opacity-40" />
            <span className="text-xs">尚未生成样本图。配置 sampling.everyNEpochs 后会自动出现</span>
          </div>
        )}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
          {enriched.map((s) => {
            const url = api.jobFileUrl(jobId, s.path)
            const name = s.path.split(/[\\/]/).pop() ?? s.path
            return (
              <button
                key={s.path}
                type="button"
                onClick={() => setOpenSrc(url)}
                className="group relative aspect-square overflow-hidden rounded-[4px] border border-border/60 bg-muted/20 transition hover:border-primary/60"
                title={name}
              >
                <img
                  src={url}
                  alt={name}
                  loading="lazy"
                  className="h-full w-full object-cover transition group-hover:scale-105"
                />
                <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 text-[10px] font-medium text-white opacity-0 transition group-hover:opacity-100">
                  {s.epoch != null && <span className="mr-2">e{s.epoch}</span>}
                  {s.step != null && <span>s{s.step}</span>}
                  {s.epoch == null && s.step == null && (
                    <span className="truncate">{name}</span>
                  )}
                </div>
              </button>
            )
          })}
        </div>
      </CardContent>
      {openSrc && <Lightbox src={openSrc} onClose={() => setOpenSrc(null)} />}
    </Card>
  )
}

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
      onClick={onClose}
    >
      <Button
        variant="ghost"
        size="sm"
        className="absolute top-4 right-4 text-white hover:bg-white/10"
        onClick={onClose}
      >
        <X className="size-4" />
      </Button>
      <img
        src={src}
        alt="sample preview"
        className="max-h-[90vh] max-w-[90vw] rounded-[4px] object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  )
}
