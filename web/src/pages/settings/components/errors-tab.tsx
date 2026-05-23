import { useMemo, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  Bug,
  Copy,
  Download,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"

import {
  errorReportsApi,
  type ErrorReportItem,
} from "@/lib/api"
import {
  getReportingEnabled,
  setReportingEnabled,
} from "@/lib/error-reporter"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type SeverityFilter = "all" | ErrorReportItem["severity"]
type SourceFilter =
  | "all"
  | "backend.exception"
  | "backend.job"
  | "backend.lifespan"
  | "backend.preflight"
  | "backend.bootstrap"
  | "backend.update"
  | "frontend.render"
  | "frontend.runtime"
  | "frontend.api"
  | "user.report"

const SEVERITY_LABEL: Record<ErrorReportItem["severity"], string> = {
  fatal: "严重",
  error: "错误",
  warn: "警告",
  info: "信息",
}

const SEVERITY_TONE: Record<ErrorReportItem["severity"], string> = {
  fatal: "text-destructive",
  error: "text-destructive",
  warn: "text-amber-700 dark:text-amber-400",
  info: "text-cyan-700 dark:text-cyan-400",
}

const SOURCE_LABEL: Record<string, string> = {
  "backend.exception": "后端 · 未捕获异常",
  "backend.job": "后端 · 训练任务",
  "backend.lifespan": "后端 · 启动钩子",
  "backend.preflight": "后端 · 预检查",
  "backend.bootstrap": "后端 · 安装",
  "backend.update": "后端 · 自更新",
  "frontend.render": "前端 · 渲染崩溃",
  "frontend.runtime": "前端 · 运行时",
  "frontend.api": "前端 · API 调用",
  "user.report": "用户主动上报",
}

export function ErrorsTab() {
  const qc = useQueryClient()
  const [severity, setSeverity] = useState<SeverityFilter>("all")
  const [source, setSource] = useState<SourceFilter>("all")
  const [q, setQ] = useState("")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [enabled, setEnabledLocal] = useState(getReportingEnabled())

  const list = useQuery({
    queryKey: ["error-reports", severity, source, q],
    queryFn: () =>
      errorReportsApi.list({
        limit: 200,
        severity: severity === "all" ? undefined : severity,
        source: source === "all" ? undefined : source,
        q: q || undefined,
      }),
    refetchInterval: 30_000,
  })

  const items = list.data?.items ?? []
  const total = list.data?.total ?? 0
  const selected = useMemo(
    () => items.find((it) => it.id === selectedId) ?? items[0] ?? null,
    [items, selectedId],
  )

  const refresh = () => qc.invalidateQueries({ queryKey: ["error-reports"] })

  const handleClear = async () => {
    if (!confirm(`确认清空 ${total} 条错误记录?此操作不可恢复。`)) return
    try {
      const { deleted } = await errorReportsApi.clear()
      toast.success(`已清空 ${deleted} 条错误记录`)
      refresh()
    } catch (e) {
      toast.error("清空失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await errorReportsApi.delete(id)
      toast.success("已删除")
      if (selectedId === id) setSelectedId(null)
      refresh()
    } catch (e) {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    }
  }

  return (
    <div className="space-y-4 max-w-[1400px]">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Bug className="size-4" />
            错误上报
          </CardTitle>
          <CardDescription>
            本地保存的错误事件清单(共 {total} 条)。包含训练失败、预检查阻断、
            前端渲染异常、未捕获异常等。所有数据仅保存在本机,导出 / 复制需要
            你主动操作。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2 items-center">
            <div className="flex items-center gap-2">
              <Switch
                id="reporting-enabled"
                checked={enabled}
                onCheckedChange={(value) => {
                  setReportingEnabled(value)
                  setEnabledLocal(value)
                  toast.message(value ? "已开启错误上报" : "已关闭前端错误上报", {
                    description: value
                      ? "未来出现的错误会被收录到这里。"
                      : "停止收集来自浏览器的错误事件;后端事件仍会记录。",
                  })
                }}
              />
              <label
                htmlFor="reporting-enabled"
                className="text-[12px] text-muted-foreground cursor-pointer select-none"
              >
                收集前端错误
              </label>
            </div>
            <div className="flex-1" />
            <Button
              size="sm"
              variant="outline"
              onClick={refresh}
              disabled={list.isFetching}
              className="gap-1.5"
            >
              <RefreshCw className={`size-3 ${list.isFetching ? "animate-spin" : ""}`} />
              刷新
            </Button>
            <Button size="sm" variant="outline" asChild>
              <a
                href={errorReportsApi.exportUrl()}
                download
                className="gap-1.5 inline-flex items-center"
              >
                <Download className="size-3" />
                导出 NDJSON
              </a>
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={handleClear}
              disabled={total === 0}
              className="gap-1.5 text-destructive hover:text-destructive"
            >
              <Trash2 className="size-3" />
              清空
            </Button>
          </div>

          <div className="flex flex-wrap gap-2 items-center">
            <Select
              value={severity}
              onValueChange={(v) => setSeverity(v as SeverityFilter)}
            >
              <SelectTrigger className="w-[140px]" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部严重程度</SelectItem>
                <SelectItem value="fatal">{SEVERITY_LABEL.fatal}</SelectItem>
                <SelectItem value="error">{SEVERITY_LABEL.error}</SelectItem>
                <SelectItem value="warn">{SEVERITY_LABEL.warn}</SelectItem>
                <SelectItem value="info">{SEVERITY_LABEL.info}</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={source}
              onValueChange={(v) => setSource(v as SourceFilter)}
            >
              <SelectTrigger className="w-[200px]" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部来源</SelectItem>
                {Object.entries(SOURCE_LABEL).map(([key, label]) => (
                  <SelectItem key={key} value={key}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="relative flex-1 min-w-[16rem]">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground" />
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="按标题或消息搜索..."
                className="pl-8"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-[420px,1fr] gap-3">
        <Card className="min-h-[400px]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">最近错误</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-2">
            {items.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                {list.isLoading ? "加载中..." : "暂无错误记录"}
              </div>
            ) : (
              <ul className="space-y-1 max-h-[640px] overflow-y-auto pr-1">
                {items.map((it) => (
                  <li key={it.id}>
                    <button
                      type="button"
                      onClick={() => setSelectedId(it.id)}
                      className={`w-full text-left rounded-[4px] px-3 py-2 hover:bg-muted/40 transition-colors ${
                        selected?.id === it.id ? "bg-muted/50 ring-1 ring-border" : ""
                      }`}
                    >
                      <div className="flex items-center gap-2 text-[11px]">
                        <Badge
                          variant="outline"
                          className={`rounded-[2px] ${SEVERITY_TONE[it.severity]}`}
                        >
                          {SEVERITY_LABEL[it.severity]}
                        </Badge>
                        <span className="text-muted-foreground">
                          {SOURCE_LABEL[it.source] ?? it.source}
                        </span>
                        <span className="ml-auto text-muted-foreground/60 font-mono">
                          {new Date(it.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="mt-1 text-[13px] font-medium truncate">
                        {it.title}
                      </div>
                      <div className="mt-0.5 text-[11px] text-muted-foreground truncate">
                        {it.message}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <DetailPanel item={selected} onDelete={handleDelete} />
      </div>
    </div>
  )
}

function DetailPanel({
  item,
  onDelete,
}: {
  item: ErrorReportItem | null
  onDelete: (id: string) => void
}) {
  if (!item) {
    return (
      <Card className="min-h-[400px]">
        <CardContent className="h-full flex items-center justify-center">
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

  return (
    <Card className="min-h-[400px]">
      <CardHeader className="pb-2">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <CardTitle className="text-sm truncate">{item.title}</CardTitle>
            <CardDescription className="font-mono text-[10px] mt-1">
              ID {item.id} · {new Date(item.timestamp).toLocaleString()}
            </CardDescription>
          </div>
          <div className="flex gap-2 shrink-0">
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
              onClick={() => onDelete(item.id)}
              className="gap-1.5 text-destructive hover:text-destructive"
            >
              <Trash2 className="size-3" />
              删除
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
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
      </CardContent>
    </Card>
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
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
