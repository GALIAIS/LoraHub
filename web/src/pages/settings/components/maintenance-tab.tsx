import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Archive, Database, HardDrive, Loader2, ShieldAlert, Terminal as TerminalIcon, Trash2 } from "lucide-react"
import { api } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
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
import { fmtBytes, fmtUnixSeconds } from "@/pages/jobs/utils"

/**
 * Storage maintenance — disk usage, archive pruning, HuggingFace cache
 * cleanup. Replaces the "ssh in and `rm -rf`" workflow with explicit
 * confirm-then-delete buttons.
 */
export function MaintenanceTab() {
  const qc = useQueryClient()
  const usage = useQuery({
    queryKey: ["storage-usage"],
    queryFn: api.storageUsage,
    staleTime: 5_000,
  })
  const archive = useQuery({
    queryKey: ["storage-archive"],
    queryFn: api.storageListArchive,
    staleTime: 5_000,
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["storage-usage"] })
    qc.invalidateQueries({ queryKey: ["storage-archive"] })
  }

  const deleteEntry = useMutation({
    mutationFn: (name: string) => api.storageDeleteArchiveEntry(name),
    onSuccess: invalidate,
  })
  const clearArchive = useMutation({
    mutationFn: () => api.storageClearArchive(),
    onSuccess: invalidate,
  })
  const clearHfCache = useMutation({
    mutationFn: () => api.storageClearHfCache(),
    onSuccess: invalidate,
  })

  const dirs = usage.data?.directories
  const fs = usage.data?.filesystem
  const archiveEntries = archive.data?.entries ?? []
  const archiveBytes = useMemo(
    () => archiveEntries.reduce((s, e) => s + e.bytes, 0),
    [archiveEntries],
  )

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <HardDrive className="size-4 text-muted-foreground" />
            磁盘占用
          </CardTitle>
          <CardDescription>
            当前工作目录所在文件系统的可用空间,与 lorahub 管理的几个常占空间目录。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {fs && (
            <div className="rounded-[4px] border border-border/60 bg-muted/20 px-4 py-3 text-sm">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="font-mono text-[12px]">{fs.path}</div>
                <div className="text-[11px] text-muted-foreground tabular-nums">
                  剩余 <span className="font-mono">{fmtBytes(fs.free_bytes)}</span>
                  {" / 总 "}
                  <span className="font-mono">{fmtBytes(fs.total_bytes)}</span>
                </div>
              </div>
              <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{
                    width: `${Math.min(
                      100,
                      (fs.used_bytes / Math.max(1, fs.total_bytes)) * 100,
                    ).toFixed(2)}%`,
                  }}
                />
              </div>
            </div>
          )}
          <div className="grid gap-2 sm:grid-cols-2">
            <DirRow label="runs/" entry={dirs?.runs} />
            <DirRow label="runs/_archive/" entry={dirs?.runs_archive} />
            <DirRow label="models/" entry={dirs?.models} />
            <DirRow label="HuggingFace 缓存" entry={dirs?.huggingface_cache} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Archive className="size-4 text-muted-foreground" />
            归档清理
          </CardTitle>
          <CardDescription>
            归档的训练任务工作区位于 <code>runs/_archive/</code>。
            训练历史会保留在数据库,删除仅清空工作区文件(checkpoint、events.jsonl 等)。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="text-[12px] text-muted-foreground">
              共 {archiveEntries.length} 项 ·{" "}
              <span className="font-mono">{fmtBytes(archiveBytes)}</span>
            </div>
            <ConfirmButton
              label="清空所有归档"
              destructive
              disabled={archiveEntries.length === 0 || clearArchive.isPending}
              loading={clearArchive.isPending}
              title="清空 runs/_archive"
              description={
                <>
                  即将永久删除 {archiveEntries.length} 个归档工作区,合计{" "}
                  <span className="font-mono">{fmtBytes(archiveBytes)}</span>。
                  此操作不可恢复。训练记录在数据库中保留。
                </>
              }
              onConfirm={() => clearArchive.mutate()}
            />
          </div>
          {archiveEntries.length === 0 ? (
            <div className="rounded-[4px] border border-dashed border-border/60 bg-muted/20 px-4 py-6 text-center text-sm text-muted-foreground">
              暂无归档。
            </div>
          ) : (
            <div className="rounded-[4px] border border-border/60 divide-y divide-border/40">
              {archiveEntries.map((e) => (
                <div
                  key={e.name}
                  className="flex items-center justify-between gap-3 px-3 py-2"
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-mono text-[12px] truncate">{e.name}</div>
                    <div className="text-[11px] text-muted-foreground tabular-nums">
                      {fmtBytes(e.bytes)} · {e.files} 文件 · {fmtUnixSeconds(e.mtime)}
                    </div>
                  </div>
                  <ConfirmButton
                    label="删除"
                    destructive
                    size="sm"
                    disabled={deleteEntry.isPending}
                    loading={
                      deleteEntry.isPending && deleteEntry.variables === e.name
                    }
                    title={`删除 ${e.name}`}
                    description={
                      <>
                        即将永久删除归档 <code>{e.name}</code>,释放{" "}
                        <span className="font-mono">{fmtBytes(e.bytes)}</span>。
                      </>
                    }
                    onConfirm={() => deleteEntry.mutate(e.name)}
                  />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database className="size-4 text-muted-foreground" />
            HuggingFace 缓存
          </CardTitle>
          <CardDescription>
            <code>~/.cache/huggingface/hub/</code> 或 <code>$HF_HOME/hub/</code> 下的下载副本。
            清空后下次拉取模型会重新下载。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {dirs?.huggingface_cache && dirs.huggingface_cache.exists ? (
            <div className="flex items-center justify-between gap-3">
              <div className="text-[12px]">
                <div className="font-mono text-muted-foreground">
                  {dirs.huggingface_cache.path}
                </div>
                <div className="text-[11px] text-muted-foreground tabular-nums mt-0.5">
                  {fmtBytes(dirs.huggingface_cache.bytes)} ·{" "}
                  {dirs.huggingface_cache.files} 文件
                </div>
              </div>
              <ConfirmButton
                label="清空缓存"
                destructive
                disabled={clearHfCache.isPending}
                loading={clearHfCache.isPending}
                title="清空 HuggingFace 缓存"
                description={
                  <>
                    即将永久删除 <code>{dirs.huggingface_cache.path}</code>,
                    释放{" "}
                    <span className="font-mono">
                      {fmtBytes(dirs.huggingface_cache.bytes)}
                    </span>
                    。下次模型下载会从头拉取。
                  </>
                }
                onConfirm={() => clearHfCache.mutate()}
              />
            </div>
          ) : (
            <div className="text-[12px] text-muted-foreground">
              未检测到 HuggingFace 缓存目录。
            </div>
          )}
        </CardContent>
      </Card>

      <TerminalSettingsCard />

      {(deleteEntry.isError || clearArchive.isError || clearHfCache.isError) && (
        <div className="text-xs text-destructive font-mono">
          {String(
            (deleteEntry.error || clearArchive.error || clearHfCache.error) as Error,
          )}
        </div>
      )}
    </div>
  )
}

function DirRow({
  label,
  entry,
}: {
  label: string
  entry:
    | { path: string | null; exists: boolean; bytes: number; files: number }
    | null
    | undefined
}) {
  if (!entry || !entry.exists) {
    return (
      <div className="rounded-[4px] border border-border/60 px-3 py-2">
        <div className="text-[12px]">{label}</div>
        <div className="text-[11px] text-muted-foreground">未创建</div>
      </div>
    )
  }
  return (
    <div className="rounded-[4px] border border-border/60 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[12px]">{label}</div>
        <Badge variant="outline" className="rounded-[2px] text-[10px] py-0 px-1.5">
          {entry.files} 文件
        </Badge>
      </div>
      <div className="text-[11px] text-muted-foreground font-mono tabular-nums truncate">
        {fmtBytes(entry.bytes)}
      </div>
    </div>
  )
}

function ConfirmButton({
  label,
  description,
  title,
  onConfirm,
  destructive = false,
  disabled,
  loading,
  size,
}: {
  label: string
  description: React.ReactNode
  title: string
  onConfirm: () => void
  destructive?: boolean
  disabled?: boolean
  loading?: boolean
  size?: "sm"
}) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <Button
        variant={destructive ? "destructive" : "outline"}
        size={size}
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        {loading ? (
          <Loader2 className="size-3 animate-spin" />
        ) : (
          <Trash2 className="size-3" />
        )}
        {label}
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            <AlertDialogDescription>{description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                setOpen(false)
                onConfirm()
              }}
            >
              <Trash2 className="size-3" /> 确认删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

/**
 * Settings panel for the in-app terminal: command-whitelist toggle and
 * per-command timeout. Both flags read from + write to ``Settings`` via
 * the standard PUT /api/settings round trip.
 */
function TerminalSettingsCard() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 10_000,
  })
  const s = settingsQuery.data?.settings as
    | (Record<string, unknown> & {
        terminal_unrestricted?: boolean
        terminal_command_timeout_s?: number
      })
    | undefined

  const unrestricted = !!s?.terminal_unrestricted
  const initialTimeout = Number(s?.terminal_command_timeout_s ?? 600)
  const [timeoutDraft, setTimeoutDraft] = useState<string>(String(initialTimeout))

  // Re-hydrate the local draft when the upstream value lands or
  // changes (e.g. another browser tab edited it).
  useMemo(() => {
    setTimeoutDraft(String(initialTimeout))
  }, [initialTimeout])

  const update = useMutation({
    mutationFn: (body: Parameters<typeof api.updateSettings>[0]) =>
      api.updateSettings(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["settings"] }),
  })

  const saveTimeout = () => {
    const n = Number.parseInt(timeoutDraft, 10)
    if (!Number.isFinite(n) || n < 5) return
    update.mutate({ terminal_command_timeout_s: n })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <TerminalIcon className="size-4 text-muted-foreground" />
          终端
        </CardTitle>
        <CardDescription>
          控制 「工具 → 终端」 页面的安全策略与运行时长上限。
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1 flex-1 min-w-0">
            <div className="text-sm font-medium flex items-center gap-2">
              自由命令模式
              {unrestricted && (
                <Badge
                  variant="outline"
                  className="rounded-[2px] text-[10px] border-amber-500/40 text-amber-600 dark:text-amber-400"
                >
                  <ShieldAlert className="size-3" />
                  已开启
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              关闭时(默认):只允许 <code className="font-mono">pip / uv / python</code> 命令,
              且 <code className="font-mono">pip</code> 自动改写为 <code className="font-mono">python -m pip</code>{" "}
              以确保命中所选后端的 venv。
              <br />
              开启时:任何命令都能执行,但你需要自己保证它的安全性。
            </p>
          </div>
          <Switch
            checked={unrestricted}
            onCheckedChange={(v) =>
              update.mutate({ terminal_unrestricted: v })
            }
            disabled={update.isPending}
          />
        </div>

        <div className="space-y-1.5">
          <Label className="text-[12px]">单条命令超时(秒)</Label>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={5}
              max={86400}
              value={timeoutDraft}
              onChange={(e) => setTimeoutDraft(e.target.value)}
              className="h-8 text-[12px] max-w-[8rem] font-mono"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-[11px]"
              onClick={saveTimeout}
              disabled={
                update.isPending ||
                !timeoutDraft ||
                Number.parseInt(timeoutDraft, 10) === initialTimeout
              }
            >
              保存
            </Button>
            <span className="text-[11px] text-muted-foreground">
              超时后会强制结束子进程。pip install 大轮子建议 ≥ 600。
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
