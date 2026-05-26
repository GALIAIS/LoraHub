/**
 * DiagnosisBanner — turns the heuristic ``/jobs/{id}/diagnose`` API into
 * an opinionated, action-oriented summary at the very top of the
 * analysis page.
 *
 * Why prominent placement: users almost always come to this page with a
 * specific worry ("why is loss flat", "is this overfitting", "did the
 * trainer crash silently"). The diagnoser already encodes the answers
 * — it was just buried behind a tab. Surfacing it here means the most
 * urgent signal hits the eye first.
 *
 * Behaviour:
 *   - Auto-runs once on mount (cheap; the backend is read-only and
 *     side-effect free).
 *   - Findings are bucketed into Data / Config / Numerical / Other to
 *     match the layered "where does the bug live" mental model
 *     (dataset → config → numerics).
 *   - Severity ordering: error > warn > info, with errors always
 *     visible while warns/infos collapse into a "show all" pill.
 *   - On API failure shows a compact retry, never blocks the page.
 */
import { useMemo } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Info,
  Loader2,
  RefreshCw,
  Stethoscope,
  XCircle,
} from "lucide-react"
import { useState } from "react"

import { api, type DiagnosisFinding, type JobDiagnosis } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

type Bucket = "data" | "config" | "numeric" | "other"

const BUCKET_LABEL: Record<Bucket, string> = {
  data: "数据集",
  config: "配置",
  numeric: "数值",
  other: "其他",
}

const BUCKET_ICON: Record<Bucket, React.ReactNode> = {
  data: <Database className="size-3" />,
  config: <Stethoscope className="size-3" />,
  numeric: <Cpu className="size-3" />,
  other: <Info className="size-3" />,
}

const SEV_ORDER: Record<DiagnosisFinding["severity"], number> = {
  error: 0,
  warn: 1,
  info: 2,
}

function classify(category: string): Bucket {
  const c = category.toLowerCase()
  if (
    c.includes("dataset") ||
    c.includes("caption") ||
    c.includes("bucket") ||
    c.includes("tag") ||
    c.includes("image")
  ) {
    return "data"
  }
  if (
    c.includes("nan") ||
    c.includes("grad") ||
    c.includes("inf") ||
    c.includes("loss") ||
    c.includes("oom") ||
    c.includes("mem")
  ) {
    return "numeric"
  }
  if (
    c.includes("config") ||
    c.includes("recipe") ||
    c.includes("lr") ||
    c.includes("optimizer") ||
    c.includes("schedule") ||
    c.includes("rank") ||
    c.includes("dim")
  ) {
    return "config"
  }
  return "other"
}

export function DiagnosisBanner({ jobId }: { jobId: string }) {
  const qc = useQueryClient()
  const diagnosis = useQuery({
    queryKey: ["job-diagnose", jobId],
    queryFn: () => api.diagnoseJob(jobId),
    staleTime: 30_000,
    retry: false,
  })

  const refresh = useMutation({
    mutationFn: () => api.diagnoseJob(jobId),
    onSuccess: (data) => {
      qc.setQueryData<JobDiagnosis>(["job-diagnose", jobId], data)
    },
  })

  const [expanded, setExpanded] = useState(false)

  const grouped = useMemo(() => {
    const data = diagnosis.data
    if (!data) return null
    const findings = [...data.findings].sort(
      (a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity],
    )
    const buckets: Record<Bucket, DiagnosisFinding[]> = {
      data: [],
      config: [],
      numeric: [],
      other: [],
    }
    for (const f of findings) buckets[classify(f.category)].push(f)
    const errors = findings.filter((f) => f.severity === "error").length
    const warns = findings.filter((f) => f.severity === "warn").length
    const infos = findings.filter((f) => f.severity === "info").length
    return { findings, buckets, errors, warns, infos }
  }, [diagnosis.data])

  if (diagnosis.isLoading) {
    return (
      <div className="rounded-[6px] border border-border/60 bg-muted/30 px-4 py-2.5 text-xs text-muted-foreground inline-flex items-center gap-2">
        <Loader2 className="size-3.5 animate-spin" />
        正在运行启发式诊断…
      </div>
    )
  }

  if (diagnosis.isError) {
    return (
      <div className="rounded-[6px] border border-amber-500/40 bg-amber-500/5 px-3.5 py-2.5 text-xs text-amber-700 dark:text-amber-300 flex items-center gap-2">
        <AlertTriangle className="size-3.5" />
        诊断失败:{" "}
        <span className="font-mono text-[11px]">
          {(diagnosis.error as Error).message}
        </span>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto h-6"
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
        >
          <RefreshCw className="size-3" /> 重试
        </Button>
      </div>
    )
  }

  if (!grouped || grouped.findings.length === 0) {
    return (
      <div className="rounded-[6px] border border-emerald-600/30 bg-emerald-600/5 px-3.5 py-2.5 text-xs text-emerald-700 dark:text-emerald-300 flex items-center gap-2 analysis-fade-in">
        <CheckCircle2 className="size-3.5" />
        启发式诊断未发现异常。
        <span className="ml-auto text-[10.5px] text-muted-foreground/80">
          {diagnosis.data?.summary || ""}
        </span>
        <Button
          size="sm"
          variant="ghost"
          className="h-6"
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
        >
          <RefreshCw
            className={cn("size-3", refresh.isPending && "animate-spin")}
          />
        </Button>
      </div>
    )
  }

  const { findings, buckets, errors, warns, infos } = grouped
  const headlineTone =
    errors > 0
      ? "error"
      : warns > 0
        ? "warn"
        : "info"

  // Always render every error inline; warns/infos hidden behind expand.
  const visible = expanded
    ? findings
    : findings.filter((f) => f.severity === "error")

  const hiddenCount = findings.length - visible.length

  return (
    <div
      className={cn(
        "rounded-[6px] border px-3.5 py-3 space-y-2.5 analysis-fade-in",
        headlineTone === "error"
          ? "border-destructive/40 bg-destructive/5"
          : headlineTone === "warn"
            ? "border-amber-500/40 bg-amber-500/5"
            : "border-sky-500/30 bg-sky-500/5",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "inline-flex size-6 items-center justify-center rounded-[3px]",
            headlineTone === "error"
              ? "bg-destructive/15 text-destructive"
              : headlineTone === "warn"
                ? "bg-amber-500/15 text-amber-700 dark:text-amber-300"
                : "bg-sky-500/15 text-sky-700 dark:text-sky-300",
          )}
        >
          {headlineTone === "error" ? (
            <XCircle className="size-3.5" />
          ) : headlineTone === "warn" ? (
            <AlertTriangle className="size-3.5" />
          ) : (
            <Info className="size-3.5" />
          )}
        </span>
        <div className="flex-1 min-w-0">
          <div className="text-[12px] font-semibold tracking-tight">
            诊断:
            {errors > 0 && (
              <span className="ml-1.5 text-destructive">{errors} 个错误</span>
            )}
            {warns > 0 && (
              <span className="ml-1.5 text-amber-700 dark:text-amber-300">
                {warns} 项警告
              </span>
            )}
            {infos > 0 && errors === 0 && warns === 0 && (
              <span className="ml-1.5 text-sky-700 dark:text-sky-300">
                {infos} 条提示
              </span>
            )}
          </div>
          {diagnosis.data?.summary && (
            <div className="text-[11px] text-muted-foreground/85 mt-0.5">
              {diagnosis.data.summary}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1">
          {(["data", "config", "numeric", "other"] as Bucket[]).map((b) => {
            const n = buckets[b].length
            if (n === 0) return null
            return (
              <span
                key={b}
                className="inline-flex items-center gap-1 rounded-[3px] bg-background/70 border border-border/60 px-1.5 py-0.5 text-[10.5px] text-muted-foreground"
                title={`${BUCKET_LABEL[b]}: ${n} 项`}
              >
                {BUCKET_ICON[b]}
                {BUCKET_LABEL[b]}
                <span className="font-mono tabular-nums text-foreground/80">{n}</span>
              </span>
            )
          })}
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-1.5"
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending}
            aria-label="重新诊断"
          >
            <RefreshCw
              className={cn("size-3", refresh.isPending && "animate-spin")}
            />
          </Button>
        </div>
      </div>

      <ul className="space-y-1.5">
        {visible.map((f, i) => (
          <FindingRow key={i} finding={f} />
        ))}
      </ul>

      {hiddenCount > 0 && !expanded && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
        >
          <ChevronDown className="size-3" /> 展开 {hiddenCount} 条更低优先级
        </button>
      )}
      {expanded && hiddenCount === 0 && (
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
        >
          <ChevronUp className="size-3" /> 收起
        </button>
      )}
    </div>
  )
}

function FindingRow({ finding }: { finding: DiagnosisFinding }) {
  const sevTone =
    finding.severity === "error"
      ? "border-l-destructive/60 bg-destructive/5"
      : finding.severity === "warn"
        ? "border-l-amber-500/60 bg-amber-500/5"
        : "border-l-sky-500/50 bg-sky-500/5"
  const SevIcon =
    finding.severity === "error"
      ? XCircle
      : finding.severity === "warn"
        ? AlertTriangle
        : Info
  const sevTextTone =
    finding.severity === "error"
      ? "text-destructive"
      : finding.severity === "warn"
        ? "text-amber-700 dark:text-amber-300"
        : "text-sky-700 dark:text-sky-300"

  return (
    <li
      className={cn(
        "rounded-[3px] border-l-2 bg-background/40 px-3 py-2 space-y-1",
        sevTone,
      )}
    >
      <div className="flex items-start gap-2">
        <SevIcon className={cn("size-3 mt-0.5 shrink-0", sevTextTone)} />
        <div className="min-w-0 flex-1">
          <div className="text-[12px] leading-snug">
            <span className="font-mono text-[10.5px] text-muted-foreground/70 mr-1.5">
              {finding.category}
            </span>
            <span className="text-foreground/95">{finding.message}</span>
          </div>
          {finding.remediation && (
            <div className="text-[11px] text-muted-foreground/85 mt-1 leading-relaxed">
              <span className="font-medium text-foreground/75">建议: </span>
              {finding.remediation}
            </div>
          )}
          {finding.evidence && (
            <details className="text-[10.5px] text-muted-foreground mt-1">
              <summary className="cursor-pointer select-none">
                证据
              </summary>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-[3px] border border-border/40 bg-background/70 px-2 py-1.5 font-mono text-[10.5px] text-foreground/75">
                {finding.evidence}
              </pre>
            </details>
          )}
        </div>
      </div>
    </li>
  )
}
