import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api, type BackendId } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { BackendStatusCard } from "./backend-status-card"

/**
 * Read-only overview of every registered backend. The current default carries
 * a Badge; "设为默认" persists the user's choice via PUT /api/settings.
 */
export function OverviewTab() {
  const qc = useQueryClient()
  const backendsQuery = useQuery({
    queryKey: ["backends"],
    queryFn: api.listBackends,
    staleTime: 10_000,
  })
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 10_000,
  })

  const setDefault = useMutation({
    mutationFn: (id: BackendId) => api.updateSettings({ default_backend: id }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backends"] })
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["health"] })
    },
  })
  const updateSettings = useMutation({
    mutationFn: api.updateSettings,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["health"] })
    },
  })

  if (!backendsQuery.data) {
    return (
      <div className="text-sm text-muted-foreground">正在加载后端清单…</div>
    )
  }

  const { backends, default: defaultId } = backendsQuery.data
  const settings = settingsQuery.data?.settings
  const defaultDescriptor = backends.find((b) => b.id === defaultId)
  const readyCount = backends.filter((b) => b.ready).length
  const dispatchMode = settings?.gpu_dispatch_mode ?? "one-job-per-gpu"
  const slots = settings?.max_concurrent_jobs ?? 1
  const numGpus = settings?.gpu_dispatch_num_gpus ?? ""
  const runtimeSlots = settingsQuery.data?.runtime?.scheduler_concurrency ?? slots
  const runtimeSlotIds = settingsQuery.data?.runtime?.scheduler_slots ?? []
  const restartRequired = Boolean(settingsQuery.data?.runtime?.settings_restart_required)
  const effectiveDistributedSlots =
    dispatchMode === "distributed" ? (settings?.gpu_dispatch_num_gpus ?? runtimeSlots) : 1
  const distributedWouldBeSingle = dispatchMode === "distributed" && effectiveDistributedSlots < 2

  return (
    <div className="space-y-5">
      <div className="rounded-[4px] border border-border/60 bg-muted/20 px-4 py-3 text-sm">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <span>
            当前默认后端:
            <code className="font-mono font-semibold text-foreground">
              {defaultDescriptor?.name ?? defaultId}
            </code>
          </span>
          <span className="text-[11px] text-muted-foreground font-mono">
            就绪 {readyCount} / {backends.length}
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground/80 mt-1">
          新建任务、命令行 <code className="text-foreground">lorahub</code> 与 Web
          训练入口在未指定后端时使用此默认值。
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">GPU 调度</CardTitle>
          <CardDescription>
            全局默认策略。训练配置里选择“跟随设置”时使用这里的值；4080 + V100
            这类异构卡会按同型号/同显存分组，默认不会混在同一个分布式任务里。
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-[12rem_1fr] md:items-center">
          <Label className="text-xs">并行训练槽位</Label>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              type="number"
              min={1}
              max={16}
              value={slots}
              onChange={(e) => {
                const value = Math.max(1, Number.parseInt(e.target.value, 10) || 1)
                updateSettings.mutate({ max_concurrent_jobs: value })
              }}
              className="h-8 w-24 font-mono text-xs"
            />
            <span className="text-xs text-muted-foreground">
              保存后重启服务生效；启动时会按实际 NVIDIA GPU 数自动夹紧。
            </span>
          </div>
          <div />
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-[3px] border border-border/70 bg-muted/25 px-2 py-1">
              运行中 slot:{" "}
              <code className="font-mono text-foreground">{runtimeSlots}</code>
            </span>
            {runtimeSlotIds.length > 0 && (
              <span className="rounded-[3px] border border-border/70 bg-muted/25 px-2 py-1">
                GPU:{" "}
                <code className="font-mono text-foreground">
                  {runtimeSlotIds.join(",")}
                </code>
              </span>
            )}
            {restartRequired && (
              <span className="rounded-[3px] border border-amber-500/35 bg-amber-500/10 px-2 py-1 text-amber-700 dark:text-amber-300">
                已保存 {slots}，当前服务仍按 {runtimeSlots} 个 slot 运行，需重启后生效。
              </span>
            )}
          </div>

          <Label className="text-xs">默认调度模式</Label>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={dispatchMode}
              onValueChange={(mode) => {
                const nextMode = mode as "one-job-per-gpu" | "distributed"
                updateSettings.mutate({
                  gpu_dispatch_mode: nextMode,
                  ...(nextMode === "distributed" && slots < 2
                    ? { max_concurrent_jobs: 2, gpu_dispatch_num_gpus: 2 }
                    : {}),
                })
              }}
              disabled={updateSettings.isPending || settingsQuery.isLoading}
            >
              <SelectTrigger className="h-8 w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="one-job-per-gpu">一任务一 GPU</SelectItem>
                <SelectItem value="distributed">单任务多 GPU</SelectItem>
              </SelectContent>
            </Select>
            {dispatchMode === "distributed" && (
              <Input
                type="number"
                min={1}
                max={16}
                value={numGpus}
                placeholder="全部槽位"
                onChange={(e) => {
                  const raw = e.target.value
                  const parsed = Math.max(1, Number.parseInt(raw, 10) || 1)
                  updateSettings.mutate({
                    gpu_dispatch_num_gpus:
                      raw === ""
                        ? null
                        : parsed,
                    ...(parsed > slots ? { max_concurrent_jobs: parsed } : {}),
                  })
                }}
                className="h-8 w-28 font-mono text-xs"
              />
            )}
            <Button
              size="sm"
              variant="outline"
              disabled={updateSettings.isPending}
              onClick={() =>
                updateSettings.mutate({
                  gpu_dispatch_mode: "one-job-per-gpu",
                  gpu_dispatch_num_gpus: null,
                })
              }
            >
              重置
            </Button>
          </div>
          {distributedWouldBeSingle && (
            <>
              <div />
              <div className="rounded-[3px] border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                当前“单任务多 GPU”只会拿到 1 个 slot。请把并行训练槽位和 GPU
                数量设为 2，并重启服务后再新建训练任务。
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        {backends.map((b) => (
          <BackendStatusCard
            key={b.id}
            descriptor={b}
            isDefault={b.id === defaultId}
            makingDefault={setDefault.isPending}
            onMakeDefault={() => setDefault.mutate(b.id)}
          />
        ))}
      </div>

      {setDefault.isError && (
        <div className="text-xs text-destructive font-mono">
          {(setDefault.error as Error).message}
        </div>
      )}
    </div>
  )
}
