import { useMemo, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bug,
  Download,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"

import { errorReportsApi, type ErrorReportItem } from "@/lib/api"
import {
  getReportingEnabled,
  setReportingEnabled,
} from "@/lib/error-reporter"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
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
import {
  UpstreamConfigCard,
  UpstreamStatusBadge,
} from "./errors-upstream-card"
import { DetailPanel } from "./errors-detail-panel"
import { SEVERITY_LABEL, SEVERITY_TONE, SOURCE_LABEL } from "./errors-labels"

type SeverityFilter = "all" | ErrorReportItem["severity"]
type ResolutionFilter = "all" | ErrorReportItem["resolution_status"]
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

export function ErrorsTab() {
  const qc = useQueryClient()
  const [severity, setSeverity] = useState<SeverityFilter>("all")
  const [resolution, setResolution] = useState<ResolutionFilter>("open")
  const [source, setSource] = useState<SourceFilter>("all")
  const [fingerprint, setFingerprint] = useState("")
  const [q, setQ] = useState("")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [enabled, setEnabledLocal] = useState(getReportingEnabled())
  // Confirm-clear dialog state. We replaced the native ``window.confirm``
  // with the project-wide AlertDialog so the destructive prompt picks
  // up the same chrome and theming as `artifacts → 删除工作区`.
  const [confirmingClear, setConfirmingClear] = useState(false)
  const [clearing, setClearing] = useState(false)

  const list = useQuery({
    queryKey: ["error-reports", severity, resolution, source, fingerprint, q],
    queryFn: () =>
      errorReportsApi.list({
        limit: 200,
        severity: severity === "all" ? undefined : severity,
        resolution_status: resolution === "all" ? undefined : resolution,
        source: source === "all" ? undefined : source,
        fingerprint: fingerprint || undefined,
        q: q || undefined,
      }),
    refetchInterval: 30_000,
  })
  const summary = useQuery({
    queryKey: ["error-reports-summary", severity, resolution, source, fingerprint, q],
    queryFn: () =>
      errorReportsApi.summary({
        severity: severity === "all" ? undefined : severity,
        resolution_status: resolution === "all" ? undefined : resolution,
        source: source === "all" ? undefined : source,
        fingerprint: fingerprint || undefined,
        q: q || undefined,
      }),
    refetchInterval: 30_000,
  })

  const items = list.data?.items ?? []
  const total = summary.data?.total ?? list.data?.total ?? 0
  const selected = useMemo(
    () => items.find((it) => it.id === selectedId) ?? items[0] ?? null,
    [items, selectedId],
  )

  const refresh = () => qc.invalidateQueries({ queryKey: ["error-reports"] })

  const handleClear = async () => {
    setClearing(true)
    try {
      const { deleted } = await errorReportsApi.clear()
      toast.success(`已清空 ${deleted} 条错误记录`)
      refresh()
    } catch (e) {
      toast.error("清空失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setClearing(false)
      setConfirmingClear(false)
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
            本地保存的错误事件清单（共 {total} 条）。包含训练失败、预检查阻断、
            前端渲染异常、未捕获异常等。所有数据仅保存在本机，导出 / 复制需要
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
              onClick={() => setConfirmingClear(true)}
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
              value={resolution}
              onValueChange={(v) => setResolution(v as ResolutionFilter)}
            >
              <SelectTrigger className="w-[140px]" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="open">未处理</SelectItem>
                <SelectItem value="resolved">已处理</SelectItem>
                <SelectItem value="ignored">已忽略</SelectItem>
                <SelectItem value="all">全部状态</SelectItem>
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
                placeholder="按标题或消息搜索…"
                className="pl-8"
              />
            </div>
            {fingerprint && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => setFingerprint("")}
                className="max-w-full gap-1.5 font-mono text-[11px]"
                title={fingerprint}
              >
                指纹: {fingerprint.slice(0, 12)}...
                <span className="text-muted-foreground">清除</span>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-2 md:grid-cols-4">
        <MetricCard label="筛选结果" value={total} />
        <MetricCard label="未处理" value={summary.data?.by_resolution.open ?? 0} />
        <MetricCard
          label="严重/错误"
          value={(summary.data?.by_severity.fatal ?? 0) + (summary.data?.by_severity.error ?? 0)}
        />
        <MetricCard label="待处理上游" value={summary.data?.upstream_attention ?? 0} />
      </div>

      {(summary.data?.duplicate_groups.length ?? 0) > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">高频重复</CardTitle>
            <CardDescription>
              按指纹聚合。优先处理重复次数高的根因。
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {summary.data?.duplicate_groups.map((group) => (
              <button
                type="button"
                key={group.fingerprint}
                onClick={() => setFingerprint(group.fingerprint)}
                className="rounded-md border border-border/60 bg-muted/20 p-3 text-left text-xs hover:bg-muted/40"
              >
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={`rounded-[2px] ${SEVERITY_TONE[group.severity]}`}
                  >
                    {SEVERITY_LABEL[group.severity]}
                  </Badge>
                  <span className="ml-auto font-mono text-muted-foreground">
                    {group.count} 次
                  </span>
                </div>
                <div className="mt-2 truncate font-medium" title={group.latest_title}>
                  {group.latest_title}
                </div>
                <div className="mt-1 truncate font-mono text-[10px] text-muted-foreground">
                  {group.fingerprint}
                </div>
              </button>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[420px,1fr] gap-3 items-stretch">
        <Card className="flex flex-col min-h-[420px] max-h-[calc(100vh-360px)]">
          <CardHeader className="pb-2 shrink-0">
            <CardTitle className="text-sm">最近错误</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-2 flex-1 min-h-0 overflow-hidden">
            {items.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                {list.isLoading ? "加载中…" : "暂无错误记录"}
              </div>
            ) : (
              <ul className="space-y-1 h-full overflow-y-auto pr-1">
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

      <AlertDialog
        open={confirmingClear}
        onOpenChange={(open) => {
          if (!clearing) setConfirmingClear(open)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>清空全部错误记录</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                即将永久删除本地的{" "}
                <code className="font-mono text-xs">{total}</code> 条错误记录,
                包括尚未发送到远端的条目。
              </span>
              <span className="block text-amber-700 dark:text-amber-400">
                ⚠ 已发送到 GitLab / Gitea / Webhook 的远端 issue 不会被一并清理,需到对应仓库自行处理。
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clearing}>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClear}
              disabled={clearing}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {clearing ? "清空中…" : "确认清空"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <CardContent className="p-3">
        <div className="text-[11px] text-muted-foreground">{label}</div>
        <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
      </CardContent>
    </Card>
  )
}
