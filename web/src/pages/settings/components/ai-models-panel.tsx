import { useEffect, useMemo, useState, type ReactNode } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Loader2,
  Plus,
  Save,
  Search,
  Trash2,
} from "lucide-react"
import {
  api,
  type AIModelRecord,
  type AIProviderRecord,
} from "@/lib/api"
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
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

export function ModelsPanel() {
  const qc = useQueryClient()
  const providers = useQuery({
    queryKey: ["ai-providers"],
    queryFn: api.aiListProviders,
    staleTime: 30_000,
  })
  const [providerFilter, setProviderFilter] = useState<string>("all")
  const models = useQuery({
    queryKey: ["ai-models", providerFilter],
    queryFn: () =>
      providerFilter === "all"
        ? api.aiListModels()
        : api.aiListModels(providerFilter),
    staleTime: 30_000,
  })
  const [adding, setAdding] = useState(false)
  const [newProviderId, setNewProviderId] = useState<string>("")
  const [newModelId, setNewModelId] = useState("")
  const [newDisplayName, setNewDisplayName] = useState("")

  const providerList = providers.data?.providers ?? []
  const list = models.data?.models ?? []

  const providerById = useMemo(() => {
    const m = new Map<string, AIProviderRecord>()
    for (const p of providerList) m.set(p.id, p)
    return m
  }, [providerList])

  const grouped = useMemo(() => {
    const groups = new Map<string, AIModelRecord[]>()
    for (const m of list) {
      const arr = groups.get(m.providerId) ?? []
      arr.push(m)
      groups.set(m.providerId, arr)
    }
    return groups
  }, [list])

  const invalidateModels = () =>
    qc.invalidateQueries({ queryKey: ["ai-models"] })

  const discover = useMutation({
    mutationFn: (providerId: string) => api.aiDiscoverModels(providerId),
    onSuccess: invalidateModels,
  })

  const addModel = useMutation({
    mutationFn: () =>
      api.aiSaveModel({
        providerId: newProviderId,
        modelId: newModelId.trim(),
        displayName: newDisplayName.trim() || newModelId.trim(),
        source: "manual",
        enabled: true,
      }),
    onSuccess: () => {
      setNewModelId("")
      setNewDisplayName("")
      setAdding(false)
      invalidateModels()
    },
  })

  const toggleEnabled = useMutation({
    mutationFn: (m: AIModelRecord) =>
      api.aiSaveModel({
        id: m.id,
        providerId: m.providerId,
        modelId: m.modelId,
        displayName: m.displayName,
        source: m.source,
        enabled: !m.enabled,
        raw: m.raw,
      }),
    onSuccess: invalidateModels,
  })

  const removeModel = useMutation({
    mutationFn: (id: string) => api.aiDeleteModel(id),
    onSuccess: invalidateModels,
  })

  useEffect(() => {
    if (!newProviderId && providerList.length > 0) {
      setNewProviderId(
        providerFilter !== "all" ? providerFilter : providerList[0].id,
      )
    }
  }, [providerList, providerFilter, newProviderId])

  if (providers.isLoading) {
    return (
      <div className="text-[12px] text-muted-foreground flex items-center gap-1.5">
        <Loader2 className="size-3 animate-spin" /> 加载中…
      </div>
    )
  }

  if (providerList.length === 0) {
    return (
      <Card className="rounded-[6px] border-dashed border-border/60">
        <CardContent className="px-6 py-12 text-center text-sm text-muted-foreground">
          先到「服务商」面板添加至少一个服务商，再来管理模型。
        </CardContent>
      </Card>
    )
  }

  const filteredProviderIds =
    providerFilter === "all"
      ? providerList.map((p) => p.id)
      : [providerFilter]

  return (
    <div className="space-y-3">
      <Card>
        <CardContent className="px-4 py-3 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={() => setProviderFilter("all")}
              className={cn(
                "rounded-[2px] border px-2 py-1 text-[11px] font-mono transition-colors",
                providerFilter === "all"
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border/60 text-muted-foreground hover:bg-muted/40",
              )}
            >
              全部 ({list.length})
            </button>
            {providerList.map((p) => {
              const count =
                providerFilter === "all"
                  ? grouped.get(p.id)?.length ?? 0
                  : list.filter((m) => m.providerId === p.id).length
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setProviderFilter(p.id)}
                  className={cn(
                    "rounded-[2px] border px-2 py-1 text-[11px] transition-colors",
                    providerFilter === p.id
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border/60 text-muted-foreground hover:bg-muted/40",
                  )}
                >
                  {p.name}
                  <span className="ml-1 text-muted-foreground/85">({count})</span>
                </button>
              )
            })}
            <span className="ml-auto" />
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-[11px]"
              onClick={() => setAdding((v) => !v)}
            >
              <Plus className="size-3" /> 手工添加
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-[11px]"
              disabled={
                providerFilter === "all" ||
                discover.isPending
              }
              onClick={() => discover.mutate(providerFilter)}
              title={
                providerFilter === "all"
                  ? "选中具体服务商后才能发现"
                  : "调用 GET /v1/models 重新拉取此服务商的模型清单"
              }
            >
              {discover.isPending ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Search className="size-3" />
              )}
              发现模型
            </Button>
          </div>
          {adding && (
            <div className="rounded-[3px] border border-border/60 p-2 space-y-1.5">
              <div className="grid grid-cols-2 gap-2">
                <Field label="服务商">
                  <Select
                    value={newProviderId}
                    onValueChange={(v) => setNewProviderId(v ?? "")}
                  >
                    <SelectTrigger className="h-8 text-[12px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {providerList.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="model id">
                  <Input
                    value={newModelId}
                    onChange={(e) => setNewModelId(e.target.value)}
                    placeholder="例: gpt-4o-mini"
                    className="h-8 font-mono text-[12px]"
                  />
                </Field>
              </div>
              <Field label="显示名 (留空用 model id)">
                <Input
                  value={newDisplayName}
                  onChange={(e) => setNewDisplayName(e.target.value)}
                  className="h-8 text-[12px]"
                />
              </Field>
              <div className="flex justify-end gap-1.5">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-[11px]"
                  onClick={() => setAdding(false)}
                >
                  取消
                </Button>
                <Button
                  size="sm"
                  className="h-7 text-[11px]"
                  disabled={
                    !newProviderId || !newModelId.trim() || addModel.isPending
                  }
                  onClick={() => addModel.mutate()}
                >
                  {addModel.isPending ? (
                    <Loader2 className="size-3 animate-spin" />
                  ) : (
                    <Save className="size-3" />
                  )}
                  添加
                </Button>
              </div>
              {addModel.isError && (
                <div className="text-[11px] font-mono text-destructive">
                  {(addModel.error as Error).message}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {discover.isError && (
        <div className="text-[11px] font-mono text-destructive">
          {(discover.error as Error).message}
        </div>
      )}

      {list.length === 0 ? (
        <Card className="rounded-[6px] border-dashed border-border/60">
          <CardContent className="px-6 py-10 text-center text-[12px] text-muted-foreground">
            {providerFilter === "all"
              ? "尚无任何模型。选一个服务商点「发现模型」自动拉取，或手工添加。"
              : "此服务商尚未导入模型。点「发现模型」从 /v1/models 拉取。"}
          </CardContent>
        </Card>
      ) : (
        filteredProviderIds.map((pid) => {
          const items = grouped.get(pid) ?? []
          if (items.length === 0) return null
          const provider = providerById.get(pid)
          return (
            <Card key={pid}>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  {provider?.name ?? pid}
                  <Badge variant="outline" className="rounded-[2px] text-[10px]">
                    {items.length}
                  </Badge>
                </CardTitle>
                {provider?.baseUrl && (
                  <CardDescription className="font-mono text-[11px] truncate">
                    {provider.baseUrl}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-1 max-h-[24rem] overflow-y-auto">
                {items.map((m) => (
                  <div
                    key={m.id}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-[3px] border border-border/40 text-[12px]"
                  >
                    <span className="font-mono">{m.modelId}</span>
                    {m.displayName !== m.modelId && (
                      <span className="text-muted-foreground truncate">
                        · {m.displayName}
                      </span>
                    )}
                    <Badge
                      variant={m.source === "discovered" ? "secondary" : "outline"}
                      className="text-[9px] rounded-[2px]"
                    >
                      {m.source === "discovered" ? "已发现" : "手动"}
                    </Badge>
                    <span className="ml-auto" />
                    <Switch
                      checked={m.enabled}
                      onCheckedChange={() => toggleEnabled.mutate(m)}
                      disabled={toggleEnabled.isPending}
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-1 text-destructive hover:text-destructive"
                      onClick={() => removeModel.mutate(m.id)}
                    >
                      <Trash2 className="size-3" />
                    </Button>
                  </div>
                ))}
              </CardContent>
            </Card>
          )
        })
      )}
    </div>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px]">{label}</Label>
      {children}
      {hint && (
        <p className="text-[10px] text-muted-foreground/85 leading-relaxed">
          {hint}
        </p>
      )}
    </div>
  )
}
