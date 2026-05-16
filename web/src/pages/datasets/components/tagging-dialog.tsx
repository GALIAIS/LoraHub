import { useEffect, useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Loader2, Sparkles } from "lucide-react"
import { api, type TagDatasetRequest } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Progress,
  ProgressIndicator,
  ProgressLabel,
  ProgressTrack,
} from "@/components/ui/progress"

type Device = "auto" | "cpu" | "cuda"
const DEVICE_OPTIONS: Array<{ value: Device; label: string }> = [
  { value: "auto", label: "自动" },
  { value: "cpu", label: "CPU" },
  { value: "cuda", label: "CUDA" },
]

export function TaggingDialog({
  open,
  onOpenChange,
  path,
  onCompleted,
}: {
  open: boolean
  onOpenChange: (next: boolean) => void
  path: string
  onCompleted: () => void
}) {
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    enabled: open,
  })
  const [device, setDevice] = useState<Device>("auto")
  const [general, setGeneral] = useState(0.35)
  const [character, setCharacter] = useState(0.85)
  const [overwrite, setOverwrite] = useState(false)
  const [recursive, setRecursive] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  useEffect(() => {
    if (settings.data) setDevice(settings.data.settings.tagger_device)
  }, [settings.data])

  useEffect(() => {
    if (!open) return
    setSessionId(null)
  }, [open])

  const start = useMutation({
    mutationFn: (body: TagDatasetRequest) => api.tagDataset(body),
    onSuccess: (session) => setSessionId(session.session_id),
  })

  const session = useQuery({
    queryKey: ["tagging-session", sessionId],
    queryFn: () => api.getTaggingSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 700 : false,
  })

  useEffect(() => {
    if (session.data?.status === "succeeded") onCompleted()
  }, [session.data?.status, onCompleted])

  const current = session.data ?? start.data ?? null
  const running = current?.status === "running" || start.isPending
  const error = (start.error as Error | undefined) ?? (session.error as Error | undefined)
  const percent = Math.max(0, Math.min(100, current?.percent ?? 0))

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),36rem)]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-4 text-muted-foreground" />
            自动标注
          </DialogTitle>
          <DialogDescription className="font-mono break-all">{path}</DialogDescription>
        </DialogHeader>

        {!current ? (
          <div className="space-y-4">
            <div className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-3 items-center">
              <Label className="text-xs">设备</Label>
              <div className="flex gap-2">
                {DEVICE_OPTIONS.map((d) => (
                  <Button
                    key={d.value}
                    type="button"
                    size="sm"
                    variant={device === d.value ? "default" : "outline"}
                    onClick={() => setDevice(d.value)}
                  >
                    {d.label}
                  </Button>
                ))}
              </div>

              <Label className="text-xs">general 阈值</Label>
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={general}
                onChange={(e) => setGeneral(clamp01(Number(e.target.value)))}
                className="font-mono w-32"
              />

              <Label className="text-xs">character 阈值</Label>
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={character}
                onChange={(e) => setCharacter(clamp01(Number(e.target.value)))}
                className="font-mono w-32"
              />

              <Label className="text-xs">已有标注</Label>
              <div className="flex gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant={overwrite ? "outline" : "default"}
                  onClick={() => setOverwrite(false)}
                >
                  跳过
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant={overwrite ? "default" : "outline"}
                  onClick={() => setOverwrite(true)}
                >
                  覆盖重写
                </Button>
              </div>

              <Label className="text-xs">递归子目录</Label>
              <div className="flex gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant={recursive ? "outline" : "default"}
                  onClick={() => setRecursive(false)}
                >
                  否
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant={recursive ? "default" : "outline"}
                  onClick={() => setRecursive(true)}
                >
                  是
                </Button>
              </div>
            </div>

            <div className="text-[11px] text-muted-foreground/80">
              使用 WD-v1.4-vit-tagger-v2，首次运行会下载约 400MB 模型。
            </div>

            {error && (
              <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
                {error.message}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
                取消
              </Button>
              <Button
                size="sm"
                onClick={() =>
                  start.mutate({
                    path,
                    device,
                    general,
                    character,
                    overwrite,
                    recursive,
                  })
                }
                disabled={start.isPending}
              >
                {start.isPending ? <Loader2 className="size-3 animate-spin" /> : <Sparkles className="size-3" />}
                开始标注
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Progress value={percent} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-2">
              <ProgressLabel className="text-xs text-muted-foreground truncate">
                {current.active_provider || (running ? "加载模型…" : "等待中")}
              </ProgressLabel>
              <div className="text-xs font-mono tabular-nums text-muted-foreground text-right min-w-[5ch]">
                {percent.toFixed(1)}%
              </div>
              <ProgressTrack className="col-span-2 h-3">
                <ProgressIndicator />
              </ProgressTrack>
            </Progress>

            <dl className="grid grid-cols-3 gap-3 text-xs">
              <Stat label="状态" value={statusLabel(current.status)} />
              <Stat label="已写入" value={`${current.written}${current.total ? ` / ${current.total}` : ""}`} />
              <Stat label="设备" value={current.active_provider || current.device} />
            </dl>

            {current.events.length ? (
              <div className="rounded-[4px] border border-border/60 bg-muted/25 max-h-56 overflow-y-auto">
                <ul className="divide-y divide-border/40">
                  {current.events.slice(-12).reverse().map((event, idx) => (
                    <li key={`${event.ts}-${idx}`} className="px-3 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-muted-foreground">
                          {new Date(event.ts * 1000).toLocaleTimeString()}
                        </span>
                        <span className="font-mono">{event.percent.toFixed(1)}%</span>
                      </div>
                      <div className="mt-0.5 text-muted-foreground break-words">{event.message}</div>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {current.status === "failed" && current.error && (
              <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
                {current.error}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSessionId(null)
                  start.reset()
                }}
                disabled={running}
              >
                重新设置
              </Button>
              <Button size="sm" onClick={() => onOpenChange(false)} disabled={running}>
                {running ? <Loader2 className="size-3 animate-spin" /> : null}
                {running ? "运行中" : "完成"}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-background/45 px-3 py-2 min-w-0">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono tabular-nums text-sm truncate" title={value}>
        {value}
      </div>
    </div>
  )
}

function statusLabel(status: "running" | "succeeded" | "failed"): string {
  if (status === "running") return "运行中"
  if (status === "succeeded") return "已完成"
  return "失败"
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0
  return Math.max(0, Math.min(1, v))
}
