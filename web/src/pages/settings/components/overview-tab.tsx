import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api, type BackendId } from "@/lib/api"
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
  })

  const setDefault = useMutation({
    mutationFn: (id: BackendId) => api.updateSettings({ default_backend: id }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backends"] })
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
  const defaultDescriptor = backends.find((b) => b.id === defaultId)
  const readyCount = backends.filter((b) => b.ready).length

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
