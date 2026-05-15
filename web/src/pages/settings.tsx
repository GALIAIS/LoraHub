import { useState, useEffect } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Save,
  RotateCcw,
  Download,
  Loader2,
} from "lucide-react"
import {
  api,
  useBootstrapStream,
  type BackendStatus,
  type BootstrapEvent,
  type SettingsState,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function SettingsPage() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const [draft, setDraft] = useState<SettingsState | null>(null)

  // Sync the editable draft whenever the server-side settings load or change.
  useEffect(() => {
    if (settingsQuery.data) {
      setDraft(settingsQuery.data.settings)
    }
  }, [settingsQuery.data])

  const update = useMutation({
    mutationFn: (patch: Partial<SettingsState>) => api.updateSettings(patch),
    onSuccess: (data) => {
      qc.setQueryData(["settings"], data)
      qc.invalidateQueries({ queryKey: ["health"] })
      setDraft(data.settings)
    },
  })

  if (!draft || !settingsQuery.data) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="px-8 py-7 text-sm text-muted-foreground">正在加载设置…</div>
      </div>
    )
  }

  const backend = settingsQuery.data.backend
  const dirty =
    draft.sd_scripts_path !== settingsQuery.data.settings.sd_scripts_path ||
    draft.python_executable !== settingsQuery.data.settings.python_executable ||
    draft.tagger_device !== settingsQuery.data.settings.tagger_device

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 max-w-[900px]">
      <header className="space-y-1">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          工作台
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">设置</h1>
        <p className="text-sm text-muted-foreground">
          工作区级别的默认值。配方文件中的同名字段会按任务覆盖；环境变量
          (LORAHUB_KOHYA_*) 优先级最高。
        </p>
      </header>

      <BackendStatusCard backend={backend} />
      <BootstrapPanel />

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Kohya 后端</CardTitle>
          <CardDescription>
            指定 kohya-ss/sd-scripts 检出目录与运行它的 Python 解释器。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Field
            label="sd-scripts 路径"
            description="kohya-ss/sd-scripts 检出目录的绝对路径。留空则使用 ./sd-scripts（项目内）或 platformdirs 用户目录。"
            value={draft.sd_scripts_path ?? ""}
            placeholder="C:\\path\\to\\sd-scripts"
            onChange={(v) => setDraft({ ...draft, sd_scripts_path: v || null })}
          />
          <Field
            label="Python 解释器"
            description="可选。默认使用 <sd-scripts>/venv/Scripts/python.exe（Windows）或 .../bin/python（Unix）。"
            value={draft.python_executable ?? ""}
            placeholder="<sd-scripts>/venv/Scripts/python.exe"
            onChange={(v) => setDraft({ ...draft, python_executable: v || null })}
          />
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">自动标注</CardTitle>
          <CardDescription>
            <code className="text-foreground">lorahub tag</code> 与 Web 标注器在未显式指定时使用的默认设备。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">WD14 设备</Label>
            <div className="flex gap-2">
              {(["auto", "cpu", "cuda"] as const).map((d) => (
                <Button
                  key={d}
                  type="button"
                  size="sm"
                  variant={draft.tagger_device === d ? "default" : "outline"}
                  onClick={() => setDraft({ ...draft, tagger_device: d })}
                >
                  {d === "auto" ? "自动" : d.toUpperCase()}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center gap-3 sticky bottom-4 bg-background/80 backdrop-blur rounded-[4px] border border-border/60 px-4 py-3 shadow-[var(--panel-shadow)]">
        <Button
          size="sm"
          disabled={!dirty || update.isPending}
          onClick={() =>
            update.mutate({
              sd_scripts_path: draft.sd_scripts_path,
              python_executable: draft.python_executable,
              tagger_device: draft.tagger_device,
            })
          }
        >
          <Save className="size-3" />
          {update.isPending ? "保存中…" : "保存"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!dirty || update.isPending}
          onClick={() => settingsQuery.data && setDraft(settingsQuery.data.settings)}
        >
          <RotateCcw className="size-3" />
          重置
        </Button>
        {update.isError && (
          <span className="text-xs text-destructive font-mono">
            {(update.error as Error).message}
          </span>
        )}
        {update.isSuccess && !dirty && (
          <span className="text-xs text-emerald-600 dark:text-emerald-400">已保存。</span>
        )}
        <span className="ml-auto text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70 font-mono truncate">
          {settingsQuery.data.path}
        </span>
      </div>
      </div>
    </div>
  )
}

function BackendStatusCard({ backend }: { backend: BackendStatus }) {
  const overall =
    backend.sd_scripts_ok && backend.python_ok
      ? "ready"
      : backend.sd_scripts_ok
        ? "no-python"
        : "broken"
  const tone = {
    ready: "border-emerald-500/40 bg-emerald-500/5",
    "no-python": "border-amber-500/40 bg-amber-500/5",
    broken: "border-destructive/40 bg-destructive/5",
  }[overall]

  return (
    <div
      className={cn(
        "rounded-[6px] border px-4 py-3 shadow-[var(--panel-shadow)] flex items-start gap-3",
        tone,
      )}
    >
      <StatusIcon ok={backend.sd_scripts_ok && backend.python_ok} warn={overall === "no-python"} />
      <div className="flex-1 min-w-0 space-y-1.5">
        <div className="text-sm font-semibold tracking-tight">
          {overall === "ready" && "后端就绪"}
          {overall === "no-python" && "找到 sd-scripts，但未检测到 Python"}
          {overall === "broken" && "后端未配置"}
        </div>
        <dl className="text-xs grid grid-cols-[8rem_1fr] gap-x-3 gap-y-0.5 font-mono">
          <dt className="text-muted-foreground">sd-scripts</dt>
          <dd className="truncate" title={backend.sd_scripts_path}>
            {backend.sd_scripts_path}
          </dd>
          <dt className="text-muted-foreground">python</dt>
          <dd className="truncate" title={backend.python ?? ""}>
            {backend.python ?? "—"}
          </dd>
          <dt className="text-muted-foreground">source</dt>
          <dd>{backend.source}</dd>
        </dl>
        {backend.missing_scripts.length > 0 && (
          <div className="text-[11px] text-destructive">
            缺失文件：{backend.missing_scripts.join(", ")}
          </div>
        )}
      </div>
    </div>
  )
}

function StatusIcon({ ok, warn }: { ok: boolean; warn: boolean }) {
  if (ok) return <CheckCircle2 className="size-5 text-emerald-600 dark:text-emerald-400 shrink-0" />
  if (warn) return <AlertTriangle className="size-5 text-amber-600 dark:text-amber-400 shrink-0" />
  return <XCircle className="size-5 text-destructive shrink-0" />
}

function Field({
  label,
  description,
  value,
  placeholder,
  onChange,
}: {
  label: string
  description: string
  value: string
  placeholder?: string
  onChange: (v: string) => void
}) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-start">
      <Label className="text-xs pt-2">{label}</Label>
      <div className="min-w-0">
        <Input
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="font-mono w-full max-w-2xl"
        />
        <p className="text-[11px] text-muted-foreground/80 mt-1">{description}</p>
      </div>
    </div>
  )
}

function BootstrapPanel() {
  const qc = useQueryClient()
  // Poll status as a fallback so the panel survives a page reload while an
  // install is mid-flight, and so we never miss a terminal frame the WS may
  // have sent before we attached.
  const statusQuery = useQuery({
    queryKey: ["backend-bootstrap-status"],
    queryFn: api.getBootstrapStatus,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
  })

  const status = statusQuery.data?.status ?? "idle"
  const isRunning = status === "running"

  // Stream live events while a session is active. Prefer the streamed events
  // when the WS has produced any (it's lower latency than the poll); otherwise
  // fall back to the buffered events the polling query carries.
  const { events: streamedEvents } = useBootstrapStream(isRunning)
  const polled = statusQuery.data?.events ?? []
  const events: BootstrapEvent[] =
    streamedEvents.length > 0 ? streamedEvents : polled

  const start = useMutation({
    mutationFn: () => api.startBootstrap({}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
    },
  })

  // When an install transitions to a terminal state, refresh the backend
  // status card so the user sees their new sd-scripts checkout immediately.
  useEffect(() => {
    if (status === "succeeded" || status === "failed") {
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["health"] })
    }
  }, [status, qc])

  const startError = start.error as Error | null
  const lastError = events.find((e) => e.level === "error")

  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">安装 Kohya 后端</CardTitle>
        <CardDescription>
          克隆 <code className="text-foreground">kohya-ss/sd-scripts</code>、创建 venv，并安装
          PyTorch、依赖和 xformers。与命令行
          <code className="text-foreground"> lorahub bootstrap-kohya</code> 等价；安装在后台运行，请保持本页打开。
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap">
          <Button
            size="sm"
            disabled={isRunning || start.isPending}
            onClick={() => start.mutate()}
          >
            {isRunning ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Download className="size-3" />
            )}
            {isRunning ? "安装中…" : "安装 Kohya 后端"}
          </Button>
          <BootstrapStatusBadge status={status} />
          {startError && (
            <span className="text-xs text-destructive font-mono truncate max-w-md">
              {startError.message}
            </span>
          )}
          {!startError && status === "failed" && lastError && (
            <span className="text-xs text-destructive font-mono truncate max-w-md">
              {lastError.message}
            </span>
          )}
        </div>
        {events.length > 0 && <BootstrapEventLog events={events} />}
      </CardContent>
    </Card>
  )
}

function BootstrapStatusBadge({ status }: { status: string }) {
  const tone =
    status === "succeeded"
      ? "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/5"
      : status === "failed"
        ? "text-destructive border-destructive/40 bg-destructive/5"
        : status === "running"
          ? "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/5"
          : "text-muted-foreground border-border/60 bg-muted/30"
  return (
    <span
      className={cn(
        "px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] rounded-[3px] border font-mono",
        tone,
      )}
    >
      {status}
    </span>
  )
}

function BootstrapEventLog({ events }: { events: BootstrapEvent[] }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-muted/30 max-h-64 overflow-y-auto">
      <ol className="divide-y divide-border/40 font-mono text-[11px]">
        {events.map((ev, idx) => (
          <li
            key={`${ev.ts}-${idx}`}
            className={cn(
              "px-3 py-1.5 flex items-start gap-2",
              ev.level === "error" && "text-destructive",
              ev.level === "done" && "text-emerald-600 dark:text-emerald-400",
            )}
          >
            <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground/70 w-12 shrink-0 pt-0.5">
              {ev.level}
            </span>
            <span className="break-all">{ev.message}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
