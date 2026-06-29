import { useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  RotateCw,
  Tag as TagIcon,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioAuditReport,
  imageStudioAuditScan,
  imageStudioAutoRotate,
  imageStudioBatchByIssue,
  type AuditIssue,
  type AuditIssueKind,
  type AuditReport,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { MiniHistogram } from "../mini-histogram"
import { QuarantinePanel } from "../quarantine-panel"
import { DedupeL1Panel, SimilarityL2Panel } from "./audit-clusters"

interface Props {
  datasetPath: string
}

const ISSUE_KIND_META: Record<AuditIssueKind, { label: string; tone: string }> = {
  corrupt:         { label: "损坏 / 无法读取",       tone: "error" },
  tiny:            { label: "短边过小（<512）",        tone: "warn" },
  exif_rotation:   { label: "EXIF 旋转未应用",       tone: "warn" },
  no_caption:      { label: "缺 caption",            tone: "warn" },
  missing_trigger: { label: "缺触发词",              tone: "warn" },
  blurry:          { label: "可能模糊（Laplacian 低）", tone: "warn" },
}

export function AuditStage({ datasetPath }: Props) {
  const [params] = useSearchParams()
  const tool = params.get("tool")

  if (tool === "dedupe-l1") {
    return <DedupeL1Panel datasetPath={datasetPath} />
  }
  if (tool === "similarity-l2") {
    return <SimilarityL2Panel datasetPath={datasetPath} />
  }
  return <AuditScanPanel datasetPath={datasetPath} />
}

function AuditScanPanel({ datasetPath }: Props) {
  const qc = useQueryClient()
  const [triggerWord, setTriggerWord] = useState("")
  const [blurCheck, setBlurCheck] = useState(true)
  const [recursive, setRecursive] = useState(true)
  const reportQuery = useQuery({
    queryKey: ["image-studio-audit-report", datasetPath],
    queryFn: () => imageStudioAuditReport(datasetPath),
    enabled: Boolean(datasetPath),
  })

  const scanMutation = useMutation({
    mutationFn: () =>
      imageStudioAuditScan({
        dataset_path: datasetPath,
        recursive,
        trigger_word: triggerWord.trim() || null,
        blur_check: blurCheck,
      }),
    onSuccess: (data) => {
      toast.success(
        `审计完成 · 扫描 ${data.image_count} 张，${data.issues.length} 项异常`,
        { description: `用时 ${data.duration_s.toFixed(1)}s` },
      )
      qc.setQueryData(["image-studio-audit-report", datasetPath], data)
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("审计失败", { description: msg })
    },
  })

  const report = reportQuery.data
  const isScanning = scanMutation.isPending
  const issuesByKind = groupIssues(report?.issues ?? [])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Top bar — scan controls */}
      <div className="flex items-center gap-3 border-b border-border/60 bg-background px-4 py-2.5 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">触发词：</span>
          <Input
            value={triggerWord}
            onChange={(e) => setTriggerWord(e.target.value)}
            placeholder="@thornsdance（可选）"
            className="h-7 w-44 text-xs"
            disabled={isScanning}
          />
        </div>
        <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground select-none">
          <Switch
            checked={recursive}
            onCheckedChange={setRecursive}
            disabled={isScanning}
          />
          递归子目录
        </label>
        <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground select-none">
          <Switch
            checked={blurCheck}
            onCheckedChange={setBlurCheck}
            disabled={isScanning}
          />
          模糊检测（慢）
        </label>
        <div className="ml-auto flex items-center gap-2">
          {report && (
            <span className="text-[11px] text-muted-foreground">
              上次扫描：{new Date(report.scanned_at).toLocaleString()}
            </span>
          )}
          <Button
            size="sm"
            onClick={() => scanMutation.mutate()}
            disabled={isScanning || !datasetPath}
            className="gap-1.5"
          >
            {isScanning ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <RefreshCw className="size-3.5" />
            )}
            {report ? "重新扫描" : "开始扫描"}
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4">
        {!report && !reportQuery.isLoading && !isScanning && (
          <EmptyState />
        )}
        {reportQuery.isLoading && (
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            <Loader2 className="size-4 animate-spin mr-2" />
            加载缓存的审计报告…
          </div>
        )}
        {report && (
          <div className="grid gap-4 lg:grid-cols-2">
            <SummaryCards report={report} />
            <IssuesPanel
              report={report}
              issuesByKind={issuesByKind}
              datasetPath={datasetPath}
              onMutated={() => {
                // After a batch action, the audit report is stale —
                // re-scan so counts/histograms reflect the new state.
                scanMutation.mutate()
              }}
              className="lg:row-span-2"
            />
            <MiniHistogram
              title="分辨率分布（长边像素）"
              buckets={report.resolution_histogram}
            />
            <MiniHistogram
              title="宽高比分布"
              buckets={report.ar_histogram}
            />
            <MiniHistogram
              title="文件大小分布"
              buckets={report.filesize_histogram}
            />
            <MiniHistogram
              title="Caption 长度分布（字符）"
              buckets={report.caption_length_histogram}
            />
            <TagVocabPanel report={report} className="lg:col-span-2" />
          </div>
        )}
      </div>

      {/* Persistent quarantine drawer at the bottom — collapsed by
          default; expands to let the user review + restore. */}
      <QuarantinePanel datasetPath={datasetPath} />
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Sub-components

function SummaryCards({ report }: { report: AuditReport }) {
  const captionPct = report.image_count
    ? Math.round((report.captioned_count / report.image_count) * 100)
    : 0
  const triggerPct =
    report.trigger_word && report.image_count
      ? Math.round((report.trigger_word_hits / report.image_count) * 100)
      : null
  const okIssues = report.issues.length === 0

  return (
    <div className="rounded-md border border-border/60 bg-card p-4">
      <div className="text-xs font-medium text-foreground mb-3 flex items-center gap-1.5">
        {okIssues ? (
          <CheckCircle2 className="size-3.5 text-emerald-600" />
        ) : (
          <AlertTriangle className="size-3.5 text-amber-600" />
        )}
        概览
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="总图数" value={report.image_count} />
        <Stat
          label="已 caption"
          value={`${report.captioned_count} (${captionPct}%)`}
          tone={captionPct >= 95 ? "good" : captionPct >= 70 ? "warn" : "bad"}
        />
        <Stat
          label="异常项"
          value={report.issues.length}
          tone={report.issues.length === 0 ? "good" : "warn"}
        />
        {report.trigger_word && (
          <Stat
            label={`触发词 ${report.trigger_word}`}
            value={`${report.trigger_word_hits} (${triggerPct}%)`}
            tone={(triggerPct ?? 0) >= 95 ? "good" : "warn"}
          />
        )}
        <Stat
          label="标签词表条目"
          value={report.tag_vocab.length}
        />
        <Stat
          label="扫描用时"
          value={`${report.duration_s.toFixed(1)} s`}
        />
      </dl>
    </div>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: number | string
  tone?: "good" | "warn" | "bad"
}) {
  const toneClass =
    tone === "good"
      ? "text-emerald-700 dark:text-emerald-400"
      : tone === "bad"
        ? "text-red-600 dark:text-red-400"
        : tone === "warn"
          ? "text-amber-700 dark:text-amber-400"
          : "text-foreground"
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className={cn("text-base font-medium tabular-nums", toneClass)}>
        {value}
      </dd>
    </div>
  )
}

function IssuesPanel({
  report,
  issuesByKind,
  datasetPath,
  onMutated,
  className,
}: {
  report: AuditReport
  issuesByKind: Record<string, AuditIssue[]>
  datasetPath: string
  onMutated: () => void
  className?: string
}) {
  const qc = useQueryClient()

  const quarantineMutation = useMutation({
    mutationFn: (issueKind: AuditIssueKind) =>
      imageStudioBatchByIssue({
        dataset_path: datasetPath,
        issue_kinds: [issueKind],
        action: "quarantine",
      }),
    onSuccess: (data, issueKind) => {
      const moved = data.result?.moved_count ?? 0
      toast.success(
        `已隔离 ${moved} 张（${ISSUE_KIND_META[issueKind]?.label ?? issueKind}）`,
        {
          description:
            "图与 caption 已移到 .workbench/quarantine/，可在 整理 阶段恢复",
        },
      )
      qc.invalidateQueries({ queryKey: ["image-studio"] })
      onMutated()
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("批量隔离失败", { description: msg })
    },
  })

  const autoRotateMutation = useMutation({
    mutationFn: (paths: string[]) =>
      imageStudioAutoRotate({
        dataset_path: datasetPath,
        paths,
      }),
    onSuccess: (data) => {
      toast.success(`已应用 EXIF 旋转：${data.rotated_count} 张`)
      qc.invalidateQueries({ queryKey: ["image-studio"] })
      onMutated()
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("自动旋转失败", { description: msg })
    },
  })

  if (report.issues.length === 0) {
    return (
      <div className={cn("rounded-md border border-emerald-600/30 bg-emerald-50 dark:bg-emerald-950/20 p-4", className)}>
        <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-sm font-medium">
          <CheckCircle2 className="size-4" />
          数据集干净 · 未发现异常
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          此次扫描覆盖损坏文件 / 极小分辨率 / EXIF 旋转 / 缺 caption /
          缺触发词 / 模糊度六个维度，均未触发。
        </p>
      </div>
    )
  }
  return (
    <div className={cn("rounded-md border border-border/60 bg-card p-4 overflow-hidden flex flex-col", className)}>
      <div className="text-xs font-medium text-foreground mb-3">
        异常列表（{report.issues.length}）
      </div>
      <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
        {Object.entries(issuesByKind).map(([kind, items]) => {
          const issueKind = kind as AuditIssueKind
          const meta = ISSUE_KIND_META[issueKind] ?? {
            label: kind,
            tone: "warn",
          }
          // Available actions per issue kind. EXIF rotation gets its
          // own "auto-rotate" since baking the orientation into pixels
          // is a non-destructive fix; everything else routes to
          // quarantine where the user can review before actually
          // committing.
          const isPending =
            quarantineMutation.isPending && quarantineMutation.variables === issueKind
          const isRotating =
            autoRotateMutation.isPending && kind === "exif_rotation"
          return (
            <details
              key={kind}
              className="rounded border border-border/40 bg-background"
              open={items.length <= 5}
            >
              <summary className="cursor-pointer px-3 py-1.5 text-xs flex items-center gap-2">
                <AlertTriangle
                  className={cn(
                    "size-3.5 shrink-0",
                    meta.tone === "error"
                      ? "text-red-600"
                      : "text-amber-600",
                  )}
                />
                <span className="font-medium">{meta.label}</span>
                <span className="text-muted-foreground tabular-nums">
                  ({items.length})
                </span>
                <div className="ml-auto flex items-center gap-1">
                  {kind === "exif_rotation" && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px] gap-1"
                      disabled={isRotating}
                      onClick={(e) => {
                        e.preventDefault()
                        autoRotateMutation.mutate(items.map((i) => i.path))
                      }}
                    >
                      {isRotating ? (
                        <Loader2 className="size-3 animate-spin" />
                      ) : (
                        <RotateCw className="size-3" />
                      )}
                      自动旋转
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-6 px-2 text-[11px] gap-1 hover:text-red-600"
                    disabled={isPending}
                    onClick={(e) => {
                      e.preventDefault()
                      if (
                        !window.confirm(
                          `把这 ${items.length} 张「${meta.label}」图移到隔离区？\n` +
                            `（图与 caption 移到 .workbench/quarantine/，可恢复）`,
                        )
                      )
                        return
                      quarantineMutation.mutate(issueKind)
                    }}
                  >
                    {isPending ? (
                      <Loader2 className="size-3 animate-spin" />
                    ) : (
                      <Trash2 className="size-3" />
                    )}
                    隔离
                  </Button>
                </div>
              </summary>
              <ul className="border-t border-border/40 max-h-48 overflow-y-auto">
                {items.slice(0, 200).map((iss, i) => (
                  <li
                    key={i}
                    className="px-3 py-1 text-[11px] font-mono text-muted-foreground border-t first:border-t-0 border-border/30 truncate"
                    title={iss.path}
                  >
                    {iss.path.split(/[\\/]/).pop()}{" "}
                    {kind === "tiny" && (
                      <span className="text-[10px]">
                        {String(iss.width)}×{String(iss.height)}
                      </span>
                    )}
                    {kind === "blurry" && (
                      <span className="text-[10px]">
                        score={String(iss.score)}
                      </span>
                    )}
                    {kind === "exif_rotation" && (
                      <span className="text-[10px]">
                        ori={String(iss.orientation)}
                      </span>
                    )}
                  </li>
                ))}
                {items.length > 200 && (
                  <li className="px-3 py-1 text-[11px] text-muted-foreground border-t border-border/30">
                    ... 余 {items.length - 200} 项
                  </li>
                )}
              </ul>
            </details>
          )
        })}
      </div>
    </div>
  )
}

function TagVocabPanel({
  report,
  className,
}: {
  report: AuditReport
  className?: string
}) {
  if (report.tag_vocab.length === 0) {
    return (
      <div className={cn("rounded-md border border-border/60 bg-card p-4", className)}>
        <div className="text-xs text-muted-foreground">
          无 caption 标签可统计
        </div>
      </div>
    )
  }
  const max = report.tag_vocab[0]?.count ?? 1
  return (
    <div className={cn("rounded-md border border-border/60 bg-card p-4", className)}>
      <div className="text-xs font-medium text-foreground mb-3 flex items-center gap-1.5">
        <TagIcon className="size-3.5" />
        标签词表(前 {report.tag_vocab.length})
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-x-4 gap-y-1">
        {report.tag_vocab.map((row) => {
          const pct = (row.count / max) * 100
          return (
            <div
              key={row.tag}
              className="relative flex items-center gap-2 text-[11px] py-0.5"
            >
              <span
                aria-hidden
                className="absolute inset-y-0 left-0 bg-primary/10 rounded-sm"
                style={{ width: `${pct}%` }}
              />
              <span className="relative flex-1 truncate" title={row.tag}>
                {row.tag}
              </span>
              <span className="relative tabular-nums text-muted-foreground">
                {row.count}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center text-sm text-muted-foreground">
      <AlertTriangle className="size-8 text-muted-foreground/40 mb-3" />
      <p>该数据集还没有审计报告。</p>
      <p className="text-xs mt-1">点击右上角"开始扫描"。</p>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Helpers

function groupIssues(issues: AuditIssue[]): Record<string, AuditIssue[]> {
  const out: Record<string, AuditIssue[]> = {}
  for (const iss of issues) {
    ;(out[iss.kind] ??= []).push(iss)
  }
  return out
}
