/**
 * Diagnostics card — surfaces the heuristic failure-mode classifier
 * as a small actionable card on the job overview tab.
 *
 * Only fetches once per (jobId, state) pair: the diagnosis is
 * deterministic over the server's events.jsonl + log tail, so React
 * Query's default cache is enough. We don't poll while running —
 * diagnosis only matters after the job finishes (or stalls).
 */
import { useQuery } from "@tanstack/react-query"
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Info,
  Loader2,
  XCircle,
} from "lucide-react"
import { api, type DiagnosisFinding } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { cn } from "@/lib/utils"

const SEVERITY_ICON = {
  error: XCircle,
  warn: AlertTriangle,
  info: Info,
} as const

const SEVERITY_TONE: Record<DiagnosisFinding["severity"], string> = {
  error: "text-destructive border-destructive/40 bg-destructive/5",
  warn: "text-amber-700 dark:text-amber-400 border-amber-500/40 bg-amber-500/5",
  info: "text-sky-700 dark:text-sky-400 border-sky-500/40 bg-sky-500/5",
}

const CATEGORY_LABEL: Record<string, string> = {
  oom: "显存不足",
  nan_loss: "Loss NaN",
  missing_module: "Python 依赖缺失",
  missing_safetensors: "模型文件缺失",
  torch_compile_fail: "torch.compile 回退",
  data_loader_corrupt: "数据集损坏",
  user_cancel: "用户中断",
  vram_pressure: "显存压力",
  unknown: "未知失败",
}

interface DiagnosticsCardProps {
  jobId: string
  /**
   * Skip fetching while the job is still healthy and running. The
   * server endpoint is cheap but every redundant call writes a noisy
   * line to the access log; gating saves clutter.
   */
  enabled: boolean
}

export function DiagnosticsCard({ jobId, enabled }: DiagnosticsCardProps) {
  const { data, isPending, error, refetch, isFetching } = useQuery({
    queryKey: ["job-diagnose", jobId],
    queryFn: () => api.diagnoseJob(jobId),
    enabled,
    staleTime: 60_000,
  })

  if (!enabled) {
    return null
  }

  if (isPending) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
            诊断
          </CardTitle>
        </CardHeader>
      </Card>
    )
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base text-destructive flex items-center gap-2">
            <XCircle className="size-4" />
            诊断失败
          </CardTitle>
          <CardDescription className="text-xs font-mono">
            {error instanceof Error ? error.message : "无法加载诊断信息"}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const findings = data.findings ?? []
  const hasFindings = findings.length > 0

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base flex items-center gap-2">
              {hasFindings ? (
                <AlertTriangle className="size-4 text-amber-600" />
              ) : (
                <CheckCircle2 className="size-4 text-emerald-600" />
              )}
              诊断
            </CardTitle>
            <CardDescription>{data.summary}</CardDescription>
          </div>
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="text-[11px] text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
            title="重新分析"
          >
            {isFetching ? <Loader2 className="size-3 animate-spin" /> : null}
            重新分析
          </button>
        </div>
      </CardHeader>

      {hasFindings && (
        <CardContent className="space-y-2">
          {findings.map((f, i) => {
            const Icon = SEVERITY_ICON[f.severity]
            return (
              <div
                key={`${f.category}-${i}`}
                className={cn(
                  "rounded-[4px] border px-3 py-2 space-y-1.5 text-xs",
                  SEVERITY_TONE[f.severity],
                )}
              >
                <div className="flex items-start gap-2">
                  <Icon className="size-3.5 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge
                        variant="outline"
                        className="rounded-[2px] text-[10px] uppercase tracking-[0.1em]"
                      >
                        {CATEGORY_LABEL[f.category] ?? f.category}
                      </Badge>
                      <span className="font-medium text-foreground">
                        {f.message}
                      </span>
                    </div>
                    <p className="mt-1.5 text-muted-foreground leading-relaxed">
                      {f.remediation}
                    </p>
                    {f.evidence && (
                      <pre className="mt-1.5 text-[10px] font-mono bg-background/60 border border-border/60 rounded-[2px] px-2 py-1 max-h-24 overflow-auto whitespace-pre-wrap break-all">
                        {f.evidence}
                      </pre>
                    )}
                  </div>
                </div>
              </div>
            )
          })}

          {data.log_path && (
            <div className="text-[11px] text-muted-foreground/80 flex items-center gap-1.5 pt-1">
              <ExternalLink className="size-3" />
              <span>训练日志:</span>
              <code className="font-mono truncate">{data.log_path}</code>
            </div>
          )}
        </CardContent>
      )}

      {!hasFindings && data.log_excerpt && (
        <CardContent className="pt-0">
          <details className="text-[11px]">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground select-none">
              查看日志尾部 (用于人工诊断)
            </summary>
            <pre className="mt-2 font-mono bg-muted/30 border border-border/60 rounded-[2px] px-2 py-1 max-h-48 overflow-auto whitespace-pre-wrap break-all">
              {data.log_excerpt}
            </pre>
          </details>
        </CardContent>
      )}
    </Card>
  )
}
