/**
 * AI analysis card — generate / re-generate / display the AI-written
 * markdown training analysis. Lead paragraph is hoisted into a hero
 * "结论" block so the user gets the punchline before scrolling the
 * full body.
 */
import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Check, Copy, Download, Loader2, Sparkles } from "lucide-react"
import { api, type JobAnalysis } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Markdown } from "@/components/markdown"

export function AICard({
  jobId,
  cached,
  loading,
  canRun,
}: {
  jobId: string
  cached: JobAnalysis | null
  loading: boolean
  canRun: boolean
}) {
  const qc = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const run = useMutation({
    mutationFn: () => api.analyzeJob(jobId),
    onSuccess: () => {
      setError(null)
      qc.invalidateQueries({ queryKey: ["job-analysis", jobId] })
    },
    onError: (e: unknown) => {
      setError(e instanceof Error ? e.message : String(e))
    },
  })

  async function copyMd() {
    if (!cached?.markdown) return
    try {
      await navigator.clipboard.writeText(cached.markdown)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  function downloadMd() {
    if (!cached?.markdown) return
    const blob = new Blob([cached.markdown], {
      type: "text/markdown;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `analysis-${jobId.slice(-8)}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const lead = extractLead(cached?.markdown)

  return (
    <Card className="rounded-[6px] border-border/60">
      <CardHeader className="py-2.5 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Sparkles className="size-3.5 text-primary shrink-0" />
          <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground shrink-0">
            AI 分析总结
          </CardTitle>
          {cached && (
            <span className="text-[10px] text-muted-foreground/70 truncate">
              {cached.model} · {new Date(cached.generated_at).toLocaleString()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {cached && (
            <>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-[11px]"
                onClick={copyMd}
                title="复制 Markdown 原文"
              >
                {copied ? (
                  <Check className="size-3 text-emerald-500" />
                ) : (
                  <Copy className="size-3" />
                )}
                {copied ? "已复制" : "复制"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-[11px]"
                onClick={downloadMd}
                title="下载 .md 文件"
              >
                <Download className="size-3" /> 下载
              </Button>
            </>
          )}
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-[11px]"
            disabled={!canRun || run.isPending}
            onClick={() => run.mutate()}
          >
            {run.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Sparkles className="size-3" />
            )}
            {cached ? "重新分析" : "生成分析"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {loading && (
          <div className="text-xs text-muted-foreground">加载中…</div>
        )}
        {!loading && !cached && !run.isPending && (
          <div className="text-xs text-muted-foreground leading-relaxed">
            点击「生成分析」让 AI 阅读 events.jsonl + 配置摘要后给出诊断（收敛趋势、过拟合判断、LR 建议、下一次实验调整）。
            {!canRun && (
              <span className="block mt-1 text-amber-600">
                训练尚未产出指标点，先等一会再分析。
              </span>
            )}
          </div>
        )}
        {run.isPending && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            正在调用模型，通常 5-15 秒…
          </div>
        )}
        {error && (
          <div className="text-xs text-destructive break-all">
            分析失败：{error}
          </div>
        )}
        {cached && (
          <div className="space-y-3">
            {lead && (
              <div className="rounded-[5px] border border-primary/20 bg-primary/5 px-3 py-2 text-[12.5px] leading-[1.55] text-foreground/90">
                <div className="text-[10px] uppercase tracking-[0.18em] text-primary/80 mb-1">
                  结论
                </div>
                {lead}
              </div>
            )}
            <Markdown source={cached.markdown} />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/**
 * Pull the first meaningful paragraph out of an AI-generated markdown
 * document so the analysis card can render a hero "结论" block above
 * the full body. We skip blank lines, headings, list items, code
 * fences, and tables so the lead is always plain prose. Returns null
 * if no such paragraph is found within the first 24 lines.
 */
function extractLead(md: string | null | undefined): string | null {
  if (!md) return null
  const lines = md.split(/\r?\n/).slice(0, 24)
  let inFence = false
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue
    if (line.startsWith("```")) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    if (
      line.startsWith("#") ||
      line.startsWith("- ") ||
      line.startsWith("* ") ||
      /^\d+\.\s/.test(line) ||
      line.startsWith("|") ||
      line.startsWith(">")
    ) {
      continue
    }
    return line
      .replace(/\*\*(.+?)\*\*/g, "$1")
      .replace(/\*(.+?)\*/g, "$1")
      .replace(/`([^`]+?)`/g, "$1")
  }
  return null
}
