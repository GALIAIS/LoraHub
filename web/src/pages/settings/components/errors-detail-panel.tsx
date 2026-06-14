import { useState, type ReactNode } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  Copy,
  ExternalLink,
  Eye,
  Send,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"

import { errorReportsApi, type ErrorReportItem } from "@/lib/api"
import { Button, buttonVariants } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { SEVERITY_LABEL, SOURCE_LABEL } from "./errors-labels"

export function DetailPanel({
  item,
  onDelete,
}: {
  item: ErrorReportItem | null
  onDelete: (id: string) => void
}) {
  const qc = useQueryClient()
  const [previewOpen, setPreviewOpen] = useState(false)
  const [sending, setSending] = useState(false)

  if (!item) {
    return (
      <Card className="flex flex-col min-h-[420px] max-h-[calc(100vh-360px)]">
        <CardContent className="flex-1 min-h-0 flex items-center justify-center">
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <AlertTriangle className="size-4" />
            选中左侧任意条目查看详情
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderTemplate = () => buildIssueTemplate(item)
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(renderTemplate())
      toast.success("已复制 Issue 模板")
    } catch (e) {
      toast.error("复制失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    }
  }
  const handleSend = async () => {
    setSending(true)
    try {
      const res = await errorReportsApi.sendNow(item.id)
      if (res.ok) {
        toast.success("已发送到远端", {
          description: res.url ?? undefined,
          duration: 8_000,
        })
      } else {
        toast.error("发送失败", {
          description: res.error ?? "(未提供详情)",
          duration: 14_000,
        })
      }
      qc.invalidateQueries({ queryKey: ["error-reports"] })
    } catch (e) {
      toast.error("发送出错", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setSending(false)
    }
  }

  return (
    <Card className="flex flex-col min-h-[420px] max-h-[calc(100vh-360px)]">
      <CardHeader className="pb-2 shrink-0">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <CardTitle className="text-sm truncate">{item.title}</CardTitle>
            <CardDescription className="font-mono text-[10px] mt-1">
              ID {item.id} · {new Date(item.timestamp).toLocaleString()}
            </CardDescription>
          </div>
          <div className="flex gap-2 shrink-0 flex-wrap justify-end">
            <Button
              size="sm"
              variant="outline"
              onClick={handleCopy}
              className="gap-1.5"
            >
              <Copy className="size-3" />
              复制 Issue 模板
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setPreviewOpen((v) => !v)}
              className="gap-1.5"
            >
              <Eye className="size-3" />
              {previewOpen ? "收起脱敏预览" : "预览将发送内容"}
            </Button>
            {item.upstream_url && (
              <a
                href={item.upstream_url}
                target="_blank"
                rel="noreferrer"
                className={buttonVariants({
                  size: "sm",
                  variant: "outline",
                  className: "gap-1.5",
                })}
              >
                <ExternalLink className="size-3" />
                打开远端
              </a>
            )}
            <Button
              size="sm"
              onClick={handleSend}
              disabled={sending}
              className="gap-1.5"
            >
              <Send className="size-3" />
              {sending ? "发送中…" : "发送到远端"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onDelete(item.id)}
              className="gap-1.5 text-destructive hover:text-destructive"
            >
              <Trash2 className="size-3" />
              删除
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 flex-1 min-h-0 overflow-y-auto">
        <div className="grid grid-cols-2 gap-3 text-[11px]">
          <Field label="严重程度" value={SEVERITY_LABEL[item.severity]} />
          <Field
            label="来源"
            value={SOURCE_LABEL[item.source] ?? item.source}
          />
          <Field label="分类" value={item.category} />
          <Field label="lorahub 版本" value={item.version || "—"} />
          <Field label="平台" value={item.platform || "—"} />
          {item.job_id && <Field label="关联任务" value={item.job_id} />}
          {item.request_path && (
            <Field label="请求路径" value={item.request_path} />
          )}
          {item.request_id && (
            <Field label="请求 ID" value={item.request_id} mono />
          )}
          {item.fingerprint && (
            <Field label="指纹" value={item.fingerprint} mono />
          )}
          {item.upstream_status && (
            <Field
              label="远端状态"
              value={
                item.sent_at
                  ? `${item.upstream_status} · ${new Date(item.sent_at).toLocaleString()}`
                  : item.upstream_status
              }
            />
          )}
          {item.upstream_error && (
            <Field label="远端错误" value={item.upstream_error} />
          )}
        </div>

        <Section title="错误消息">
          <pre className="rounded-[4px] bg-muted/40 px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words max-h-[200px] overflow-auto">
            {item.message}
          </pre>
        </Section>

        {item.stack && (
          <Section title="堆栈">
            <pre className="rounded-[4px] bg-muted/40 px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words max-h-[260px] overflow-auto">
              {item.stack}
            </pre>
          </Section>
        )}

        {Object.keys(item.context ?? {}).length > 0 && (
          <Section title="上下文">
            <pre className="rounded-[4px] bg-muted/40 px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words max-h-[260px] overflow-auto">
              {JSON.stringify(item.context, null, 2)}
            </pre>
          </Section>
        )}

        {previewOpen && <UpstreamPreviewPane reportId={item.id} />}
      </CardContent>
    </Card>
  )
}

function UpstreamPreviewPane({ reportId }: { reportId: string }) {
  const preview = useQuery({
    queryKey: ["error-reports-preview", reportId],
    queryFn: () => errorReportsApi.upstreamPreview(reportId),
    staleTime: 30_000,
  })
  return (
    <Section title="将发送到远端的内容(已脱敏)">
      {preview.isLoading ? (
        <div className="text-[11px] text-muted-foreground">生成预览中…</div>
      ) : preview.isError ? (
        <div className="text-[11px] text-destructive">
          {(preview.error as Error).message}
        </div>
      ) : preview.data ? (
        <>
          <div className="text-[10px] text-muted-foreground mb-1">
            指纹 <span className="font-mono">{preview.data.fingerprint}</span>
            (同指纹会聚合到同一个远端 issue)
          </div>
          <pre className="rounded-[4px] bg-muted/40 px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words max-h-[280px] overflow-auto">
            {JSON.stringify(preview.data.body, null, 2)}
          </pre>
        </>
      ) : null}
    </Section>
  )
}

function Field({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div
        className={`mt-0.5 truncate ${mono ? "font-mono" : ""} text-foreground/90`}
        title={value}
      >
        {value}
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1">
        {title}
      </div>
      {children}
    </div>
  )
}

function buildIssueTemplate(it: ErrorReportItem): string {
  const lines: string[] = []
  lines.push(`### LoraHub 错误上报`)
  lines.push("")
  lines.push(`- **标题**: ${it.title}`)
  lines.push(`- **时间**: ${new Date(it.timestamp).toLocaleString()}`)
  lines.push(`- **严重程度**: ${SEVERITY_LABEL[it.severity]}`)
  lines.push(`- **来源**: ${SOURCE_LABEL[it.source] ?? it.source}`)
  lines.push(`- **分类**: ${it.category}`)
  lines.push(`- **版本**: ${it.version}`)
  lines.push(`- **平台**: ${it.platform}`)
  if (it.job_id) lines.push(`- **任务 ID**: ${it.job_id}`)
  if (it.request_path) lines.push(`- **请求路径**: ${it.request_path}`)
  if (it.request_id) lines.push(`- **请求 ID**: ${it.request_id}`)
  lines.push(`- **错误 ID**: ${it.id}`)
  lines.push("")
  lines.push("#### 错误消息")
  lines.push("```")
  lines.push(it.message)
  lines.push("```")
  if (it.stack) {
    lines.push("")
    lines.push("#### 堆栈")
    lines.push("```")
    lines.push(it.stack)
    lines.push("```")
  }
  if (Object.keys(it.context ?? {}).length > 0) {
    lines.push("")
    lines.push("#### 上下文")
    lines.push("```json")
    lines.push(JSON.stringify(it.context, null, 2))
    lines.push("```")
  }
  lines.push("")
  lines.push("#### 复现步骤")
  lines.push("1. ")
  lines.push("2. ")
  lines.push("")
  lines.push("#### 期望行为")
  lines.push("")
  return lines.join("\n")
}
