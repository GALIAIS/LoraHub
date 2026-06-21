import { useMemo, useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  Clipboard,
  FileImage,
  HardDrive,
  Loader2,
  Search,
  ServerCog,
  XCircle,
} from "lucide-react"
import type { TrainingEvent } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import {
  buildOperations,
  buildSummary,
  FILTERS,
  filterOperation,
  fmtClock,
  SEVERITY_LABEL,
  STAGE_LABEL,
  type OperationEvent,
  type OperationFilterId,
  type OperationSeverity,
} from "./event-operations-model"

type HistoryStatus = "idle" | "loading" | "ready" | "error"

function severityClass(severity: OperationSeverity): string {
  if (severity === "critical") return "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300"
  if (severity === "warning") return "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"
  if (severity === "success") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
  return "border-border bg-muted/50 text-muted-foreground"
}

function severityIcon(severity: OperationSeverity) {
  if (severity === "critical") return XCircle
  if (severity === "warning") return AlertTriangle
  if (severity === "success") return CheckCircle2
  return ServerCog
}

async function copyText(text: string) {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text)
  }
}

function connectionText(status: "idle" | "open" | "closed"): string {
  if (status === "open") return "SSE 已连接"
  if (status === "closed") return "SSE 已断开"
  return "等待连接"
}

function historyText(status: HistoryStatus): string {
  if (status === "loading") return "历史加载中"
  if (status === "error") return "历史加载失败"
  if (status === "ready") return "历史已加载"
  return "等待历史"
}

function percentText(value: number | null): string {
  if (value === null) return "-"
  if (value >= 99.95) return "100%"
  return `${value.toFixed(1)}%`
}

export function EventOperations({
  events,
  status,
  historyStatus = "idle",
  jobId,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  status: "idle" | "open" | "closed"
  historyStatus?: HistoryStatus
  jobId: string | null
  fallbackTotalSteps?: number | null
}) {
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<OperationFilterId>("all")
  const [selected, setSelected] = useState<OperationEvent | null>(null)

  const operations = useMemo(
    () => buildOperations(events, fallbackTotalSteps),
    [events, fallbackTotalSteps],
  )
  const summary = useMemo(
    () => buildSummary(operations, events, fallbackTotalSteps),
    [operations, events, fallbackTotalSteps],
  )
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return operations.filter((op) => {
      if (!filterOperation(op, filter)) return false
      if (!q) return true
      return [
        op.title,
        op.summary,
        op.status,
        STAGE_LABEL[op.stage],
        SEVERITY_LABEL[op.severity],
        op.artifactPath ?? "",
        JSON.stringify(op.event.payload),
      ]
        .join(" ")
        .toLowerCase()
        .includes(q)
    })
  }, [operations, filter, query])

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[6px] border border-border/60 bg-background">
      <div className="shrink-0 border-b border-border/60 bg-muted/25 p-3">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
          <div className="min-w-0 flex-1 rounded-[6px] border border-border/60 bg-background/80 px-3 py-2">
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <span className="shrink-0 text-xs font-medium text-foreground">
                  {summary.latestStage ? STAGE_LABEL[summary.latestStage] : "等待事件"}
                </span>
                {historyStatus === "loading" && (
                  <Loader2 className="size-3.5 shrink-0 animate-spin text-muted-foreground" />
                )}
                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                  {summary.step ?? "-"} / {summary.totalSteps ?? "?"}
                </span>
                <span className="text-xs text-muted-foreground">
                  {percentText(summary.progressPercent)}
                </span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {summary.loss !== null ? `loss ${summary.loss.toFixed(4)}` : "loss -"}
                </span>
              </div>
              <Badge variant="outline" className="shrink-0 normal-case tracking-normal">
                {connectionText(status)}
              </Badge>
            </div>
            <Progress
              value={summary.progressPercent ?? 0}
              className="mt-2 [&_[data-slot=progress-track]]:h-1.5 [&_[data-slot=progress-track]]:rounded-full [&_[data-slot=progress-track]]:border-border/70 [&_[data-slot=progress-indicator]]:rounded-full"
            />
            {summary.latestSummary && (
              <div className="mt-2 truncate text-xs text-muted-foreground">
                {summary.latestSummary}
              </div>
            )}
          </div>

          <div className="no-scrollbar flex gap-2 overflow-x-auto xl:max-w-[720px]">
            <SummaryTile label="严重" value={String(summary.critical)} tone={summary.critical ? "critical" : "normal"} />
            <SummaryTile label="关注" value={String(summary.warning)} tone={summary.warning ? "warning" : "normal"} />
            <SummaryTile label="检查点" value={String(summary.checkpoints)} />
            <SummaryTile label="采样" value={String(summary.samples)} />
            <SummaryTile label="验证" value={String(summary.validations)} />
            <SummaryTile label="诊断" value={String(summary.diagnostics)} tone={summary.diagnostics ? "warning" : "normal"} />
            <SummaryTile label="事件" value={String(summary.eventCount)} />
            <SummaryTile label="历史" value={historyText(historyStatus)} tone={historyStatus === "error" ? "critical" : "normal"} />
          </div>
        </div>

        <div className="mt-3 flex flex-col gap-2 lg:flex-row lg:items-center">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索事件、路径、错误或 payload"
              className="h-8"
            />
          </div>
          <div className="no-scrollbar flex gap-1 overflow-x-auto">
            {FILTERS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setFilter(item.id)}
                className={cn(
                  "h-8 rounded-[6px] border px-2.5 text-xs transition-colors",
                  filter === item.id
                    ? "border-primary/40 bg-primary/12 text-foreground"
                    : "border-border/60 bg-background text-muted-foreground hover:text-foreground",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <ScrollArea className="h-0 min-h-0 flex-1">
        {rows.length === 0 ? (
          <div className="flex min-h-[220px] items-center justify-center px-4 text-center text-sm text-muted-foreground">
            {events.length === 0 ? "尚未收到训练事件。" : "没有匹配的事件。"}
          </div>
        ) : (
          <>
            <div className="hidden md:block">
              <Table>
                <TableHeader className="sticky top-0 z-10">
                  <TableRow>
                    <TableHead className="w-[96px]">时间</TableHead>
                    <TableHead className="w-[96px]">级别</TableHead>
                    <TableHead className="w-[80px]">阶段</TableHead>
                    <TableHead>事件</TableHead>
                    <TableHead className="w-[104px]">步/回合</TableHead>
                    <TableHead className="w-[92px]">状态</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((op) => (
                    <OperationRow
                      key={op.id}
                      op={op}
                      onSelect={() => setSelected(op)}
                    />
                  ))}
                </TableBody>
              </Table>
            </div>
            <div className="grid gap-2 p-3 md:hidden">
              {rows.map((op) => (
                <OperationCard
                  key={op.id}
                  op={op}
                  onSelect={() => setSelected(op)}
                />
              ))}
            </div>
          </>
        )}
      </ScrollArea>

      <Sheet open={selected !== null} onOpenChange={(open) => !open && setSelected(null)}>
        <SheetContent className="w-[94vw] gap-0 p-0 sm:max-w-[640px]">
          {selected && (
            <>
              <SheetHeader>
                <div className="flex items-start gap-3 pr-9">
                  <SeverityMark severity={selected.severity} />
                  <div className="min-w-0">
                    <SheetTitle>{selected.title}</SheetTitle>
                    <SheetDescription>
                      {fmtClock(selected.event.timestamp)} · {STAGE_LABEL[selected.stage]} · {selected.status}
                    </SheetDescription>
                  </div>
                </div>
              </SheetHeader>
              <ScrollArea className="min-h-0 flex-1">
                <div className="space-y-4 p-4">
                  <div className="rounded-[6px] border border-border/60 bg-muted/25 p-3">
                    <div className="mb-1 text-xs font-medium text-muted-foreground">摘要</div>
                    <div className="break-words text-sm leading-relaxed text-foreground">
                      {selected.summary || "-"}
                    </div>
                  </div>

                  {selected.artifactPath && (
                    <div className="rounded-[6px] border border-border/60 p-3">
                      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                        {selected.event.type === "sample_ready" ? (
                          <FileImage className="size-3.5" />
                        ) : (
                          <HardDrive className="size-3.5" />
                        )}
                        产物
                      </div>
                      <div className="break-all font-mono text-xs text-foreground">
                        {selected.artifactPath}
                      </div>
                      {jobId && selected.event.type === "sample_ready" && (
                        <a
                          href={`/api/jobs/${jobId}/files/raw?path=${encodeURIComponent(selected.artifactPath)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-3 block max-w-[280px]"
                        >
                          <img
                            src={`/api/jobs/${jobId}/files/raw?path=${encodeURIComponent(selected.artifactPath)}`}
                            alt={selected.artifactPath}
                            className="max-h-[220px] rounded-[6px] border border-border/60 bg-muted object-contain"
                            loading="lazy"
                          />
                        </a>
                      )}
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <DetailTile label="事件类型" value={selected.event.type} />
                    <DetailTile label="事件序号" value={String(selected.index)} />
                    <DetailTile label="Step" value={selected.step !== null ? String(selected.step) : "-"} />
                    <DetailTile label="Epoch" value={selected.epoch !== null ? String(selected.epoch) : "-"} />
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="text-xs font-medium text-muted-foreground">原始 payload</div>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        onClick={() => void copyText(JSON.stringify(selected.event, null, 2))}
                      >
                        <Clipboard className="size-3" />
                        复制 JSON
                      </Button>
                    </div>
                    <pre className="max-h-[340px] overflow-auto rounded-[6px] border border-border/60 bg-muted/30 p-3 text-xs leading-relaxed text-foreground">
                      {JSON.stringify(selected.event.payload, null, 2)}
                    </pre>
                  </div>
                </div>
              </ScrollArea>
            </>
          )}
        </SheetContent>
      </Sheet>
    </div>
  )
}

function OperationRow({ op, onSelect }: { op: OperationEvent; onSelect: () => void }) {
  const Icon = severityIcon(op.severity)
  return (
    <TableRow className="cursor-pointer" onClick={onSelect}>
      <TableCell className="font-mono text-xs tabular-nums text-muted-foreground">
        {fmtClock(op.event.timestamp)}
      </TableCell>
      <TableCell>
        <Badge
          variant="outline"
          className={cn("gap-1.5 normal-case tracking-normal", severityClass(op.severity))}
        >
          <Icon className="size-3" />
          {SEVERITY_LABEL[op.severity]}
        </Badge>
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">
        {STAGE_LABEL[op.stage]}
      </TableCell>
      <TableCell className="max-w-[460px]">
        <div className="truncate text-sm font-medium text-foreground">
          {op.title}
        </div>
        <div className="truncate text-xs text-muted-foreground">
          {op.summary || "-"}
        </div>
      </TableCell>
      <TableCell className="font-mono text-xs tabular-nums text-muted-foreground">
        {op.step ?? "-"} / {op.epoch ?? "-"}
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">
        {op.status}
      </TableCell>
    </TableRow>
  )
}

function OperationCard({ op, onSelect }: { op: OperationEvent; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="rounded-[6px] border border-border/60 bg-background p-3 text-left transition-colors hover:bg-muted/35"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-mono tabular-nums">{fmtClock(op.event.timestamp)}</span>
            <span>{STAGE_LABEL[op.stage]}</span>
            <span>{op.status}</span>
          </div>
          <div className="mt-1 truncate text-sm font-medium text-foreground">{op.title}</div>
        </div>
        <SeverityMark severity={op.severity} />
      </div>
      <div className="mt-2 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
        {op.summary || "-"}
      </div>
      <div className="mt-3 font-mono text-[11px] text-muted-foreground">
        step {op.step ?? "-"} / epoch {op.epoch ?? "-"}
      </div>
    </button>
  )
}

function SeverityMark({ severity }: { severity: OperationSeverity }) {
  const Icon = severityIcon(severity)
  return (
    <span
      className={cn(
        "inline-flex size-7 shrink-0 items-center justify-center rounded-[6px] border",
        severityClass(severity),
      )}
    >
      <Icon className="size-3.5" />
    </span>
  )
}

function SummaryTile({
  label,
  value,
  tone = "normal",
}: {
  label: string
  value: string
  tone?: "normal" | "warning" | "critical"
}) {
  return (
    <div className="min-w-[76px] rounded-[6px] border border-border/60 bg-background/80 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "mt-0.5 truncate text-sm font-medium tabular-nums",
          tone === "critical" && "text-red-600 dark:text-red-300",
          tone === "warning" && "text-amber-700 dark:text-amber-300",
        )}
      >
        {value}
      </div>
    </div>
  )
}

function DetailTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-[6px] border border-border/60 bg-background px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 truncate font-mono text-xs text-foreground">{value}</div>
    </div>
  )
}
