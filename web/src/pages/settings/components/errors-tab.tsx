import { useEffect, useMemo, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  Bug,
  Copy,
  Download,
  ExternalLink,
  Eye,
  Network,
  RefreshCw,
  Search,
  Send,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"

import {
  errorReportsApi,
  type ErrorReportItem,
  type SettingsState,
  api,
} from "@/lib/api"
import {
  getReportingEnabled,
  setReportingEnabled,
} from "@/lib/error-reporter"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
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
      <UpstreamConfigCard />
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
            <a
              href={errorReportsApi.exportUrl()}
              download
              className={buttonVariants({
                size: "sm",
                variant: "outline",
                className: "gap-1.5 inline-flex items-center",
              })}
            >
              <Download className="size-3" />
              导出 NDJSON
            </a>
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
                        {it.upstream_status && (
                          <UpstreamStatusBadge status={it.upstream_status} />
                        )}
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
  const qc = useQueryClient()
  const [previewOpen, setPreviewOpen] = useState(false)
  const [sending, setSending] = useState(false)

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
    <Card className="min-h-[400px]">
      <CardHeader className="pb-2">
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
              {sending ? "发送中..." : "发送到远端"}
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
        <div className="text-[11px] text-muted-foreground">生成预览中...</div>
      ) : preview.isError ? (
        <div className="text-[11px] text-destructive">
          {(preview.error as Error).message}
        </div>
      ) : preview.data ? (
        <>
          <div className="text-[10px] text-muted-foreground mb-1">
            指纹 <span className="font-mono">{preview.data.fingerprint}</span>(同指纹会聚合到同一个远端 issue)
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

// ----------------------------------------------------------------------- //
// Upstream sink config card
//
// All four channels (off / gitlab / webhook) plus the auto-send threshold
// live here. Saving the form posts a settings patch — the backend's
// dispatcher closure re-reads settings on the next attempt, so a token
// rotation takes effect without restarting the daemon.
// ----------------------------------------------------------------------- //

type UpstreamChannel = SettingsState["error_upstream_channel"]

function UpstreamStatusBadge({ status }: { status: string }) {
  const tone =
    status === "sent"
      ? "text-emerald-600 dark:text-emerald-400"
      : status === "failed"
        ? "text-destructive"
        : "text-cyan-700 dark:text-cyan-400"
  const labelMap: Record<string, string> = {
    queued: "排队中",
    retrying: "重试中",
    sent: "已发送",
    failed: "发送失败",
    skipped: "已跳过",
  }
  return (
    <Badge variant="outline" className={`rounded-[2px] ${tone}`}>
      {labelMap[status] ?? status}
    </Badge>
  )
}

function UpstreamConfigCard() {
  const qc = useQueryClient()
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: () => api.getSettings(),
    staleTime: 30_000,
  })
  const cfg = settings.data?.settings

  const [draft, setDraft] = useState({
    error_upstream_channel: "off" as UpstreamChannel,
    error_upstream_gitlab_base_url: "",
    error_upstream_gitlab_repo: "",
    error_upstream_gitlab_token: "",
    error_upstream_webhook_url: "",
    error_upstream_webhook_auth_header: "",
    error_upstream_auto_severity: "error" as SettingsState["error_upstream_auto_severity"],
  })
  const [saving, setSaving] = useState(false)
  const [probing, setProbing] = useState(false)

  // Hydrate the draft once the settings query resolves; further user
  // edits stay local until "保存" is pressed so partial typing doesn't
  // clobber the live config.
  useEffect(() => {
    if (!cfg) return
    setDraft({
      error_upstream_channel: cfg.error_upstream_channel ?? "off",
      error_upstream_gitlab_base_url: cfg.error_upstream_gitlab_base_url ?? "",
      error_upstream_gitlab_repo: cfg.error_upstream_gitlab_repo ?? "",
      error_upstream_gitlab_token: cfg.error_upstream_gitlab_token ?? "",
      error_upstream_webhook_url: cfg.error_upstream_webhook_url ?? "",
      error_upstream_webhook_auth_header:
        cfg.error_upstream_webhook_auth_header ?? "",
      error_upstream_auto_severity: cfg.error_upstream_auto_severity ?? "error",
    })
  }, [cfg])

  const onSave = async () => {
    setSaving(true)
    try {
      await api.updateSettings(draft as Partial<SettingsState>)
      toast.success("已保存远端上报配置")
      qc.invalidateQueries({ queryKey: ["settings"] })
    } catch (e) {
      toast.error("保存失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setSaving(false)
    }
  }

  const onProbe = async () => {
    setProbing(true)
    try {
      const res = await errorReportsApi.upstreamHealth()
      if (res.ok) {
        toast.success("远端连通正常", {
          description: res.url ?? `channel=${res.channel}`,
        })
      } else {
        toast.error("远端连通失败", {
          description: res.error ?? "(未提供详情)",
          duration: 14_000,
        })
      }
    } catch (e) {
      toast.error("连通测试出错", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setProbing(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Network className="size-4" />
          远端上报通道
        </CardTitle>
        <CardDescription>
          可选,默认关闭。开启后,error 及以上的错误会自动推送到所选通道(warn/info 仍需要手动点「发送到远端」)。
          上传前会脱敏:Authorization / API key、用户主目录与盘符、邮箱与 IP 地址都会被替换。
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1">
              通道
            </div>
            <Select
              value={draft.error_upstream_channel}
              onValueChange={(v) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_channel: v as UpstreamChannel,
                }))
              }
            >
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">关闭(仅本地)</SelectItem>
                <SelectItem value="gitea">Gitea Issues (git.galiais.com 默认)</SelectItem>
                <SelectItem value="gitlab">GitLab Issues</SelectItem>
                <SelectItem value="webhook">Webhook</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1">
              自动发送阈值
            </div>
            <Select
              value={draft.error_upstream_auto_severity}
              onValueChange={(v) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_auto_severity:
                    v as SettingsState["error_upstream_auto_severity"],
                }))
              }
            >
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">全部手动</SelectItem>
                <SelectItem value="error">error 及以上自动发送</SelectItem>
                <SelectItem value="all">全部自动发送</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {(draft.error_upstream_channel === "gitlab" ||
          draft.error_upstream_channel === "gitea") && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Input
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "Gitea Base URL  https://git.galiais.com"
                  : "GitLab Base URL  https://gitlab.example.com"
              }
              value={draft.error_upstream_gitlab_base_url}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_base_url: e.target.value,
                }))
              }
            />
            <Input
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "项目路径  Shiro/LoraHubReport"
                  : "项目路径  group/project"
              }
              value={draft.error_upstream_gitlab_repo}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_repo: e.target.value,
                }))
              }
            />
            <Input
              type="password"
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "Gitea Personal Access Token (write:issue scope)"
                  : "GitLab Personal Access Token (api scope)"
              }
              value={draft.error_upstream_gitlab_token}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_token: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
            <p className="md:col-span-2 text-[11px] text-muted-foreground">
              提示:留空则回退到环境变量(
              {draft.error_upstream_channel === "gitea"
                ? "LORAHUB_GITEA_TOKEN"
                : "LORAHUB_GITLAB_TOKEN"}
              ,或通用 LORAHUB_REPORT_TOKEN)。这样 settings.json 不会留下明文 token。
            </p>
          </div>
        )}
        {draft.error_upstream_channel === "webhook" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Input
              placeholder="Webhook URL  https://hooks.example.com/lorahub"
              value={draft.error_upstream_webhook_url}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_webhook_url: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
            <Input
              type="password"
              placeholder="Authorization 头(可选,如 Bearer xxx)"
              value={draft.error_upstream_webhook_auth_header}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_webhook_auth_header: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <Button
            size="sm"
            variant="outline"
            onClick={onProbe}
            disabled={
              probing ||
              draft.error_upstream_channel === "off" ||
              !cfg
            }
            className="gap-1.5"
          >
            <Network className="size-3" />
            {probing ? "测试中..." : "测试连通"}
          </Button>
          <Button
            size="sm"
            onClick={onSave}
            disabled={saving || !cfg}
            className="gap-1.5"
          >
            {saving ? "保存中..." : "保存"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
