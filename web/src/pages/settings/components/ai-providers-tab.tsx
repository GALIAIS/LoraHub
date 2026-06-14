import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bot,
  Layers,
  Loader2,
  Plus,
  Save,
  Search,
  Settings2,
  Trash2,
} from "lucide-react"
import {
  api,
  AI_TASK_IDS,
  type AIModelRecord,
  type AIProviderRecord,
  type AIRouteRecord,
  type AITaskId,
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
import { ProvidersPanel } from "./ai-providers-panel"

const TASK_LABELS: Record<AITaskId, string> = {
  "global.default": "默认 (兜底)",
  "tagging.assist": "VLM 补打标签",
  "caption.rewrite": "Caption 改写",
  "dataset.analyze": "数据集分析",
  "training.diagnose": "训练诊断",
  "error.diagnose": "错误自助",
  "quality.score": "图片质量评分",
  "trigger.suggest": "Trigger 建议",
}

const TASK_DESCRIPTIONS: Record<AITaskId, string> = {
  "global.default": "其它任务未单独配置时的兜底路由",
  "tagging.assist": "用 VLM 给图补充 wd14 不擅长的描述 (光照、角度、自然语言)",
  "caption.rewrite": "把 wd14 标签改写为自然语言或统一格式",
  "dataset.analyze": "对扫描结果做诊断 — caption 长度、tag 分布等",
  "training.diagnose": "解读 loss/grad_norm 曲线给优化建议",
  "error.diagnose": "训练 / 安装失败时给出修复建议",
  "quality.score": "VLM 评估图片质量 (0-100 + 优/中/差)",
  "trigger.suggest": "根据数据集特征建议 trigger word 和模板",
}

const REASONING_EFFORTS = ["low", "medium", "high"] as const

export function AIProvidersTab() {
  const [activePanel, setActivePanel] = useState<
    "providers" | "models" | "routes"
  >("providers")
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant={activePanel === "providers" ? "default" : "outline"}
          onClick={() => setActivePanel("providers")}
          className="h-8"
        >
          <Settings2 className="size-3.5" /> 服务商
        </Button>
        <Button
          size="sm"
          variant={activePanel === "models" ? "default" : "outline"}
          onClick={() => setActivePanel("models")}
          className="h-8"
        >
          <Bot className="size-3.5" /> 模型
        </Button>
        <Button
          size="sm"
          variant={activePanel === "routes" ? "default" : "outline"}
          onClick={() => setActivePanel("routes")}
          className="h-8"
        >
          <Layers className="size-3.5" /> 任务路由
        </Button>
      </div>
      {activePanel === "providers" && <ProvidersPanel />}
      {activePanel === "models" && <ModelsPanel />}
      {activePanel === "routes" && <RoutesPanel />}
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
  children: React.ReactNode
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

// --------------------------------------------------------------------------- //
// Models panel: standalone CRUD + provider filter, sibling to providers/routes
// --------------------------------------------------------------------------- //

function ModelsPanel() {
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

  // Group models by their provider id (preserving display order).
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

  // Pick a provider by default for the "manual add" form so the
  // provider-id Select isn't blank on open.
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
                {/* OpenRouter / 中转站「发现模型」一键导入后,单 provider
                    可能上百行 — 不限高的话整个面板被推到屏幕下方。 */}
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

// --------------------------------------------------------------------------- //
// Routes panel: per-task (provider + model + system prompt + sampling)
// --------------------------------------------------------------------------- //

function RoutesPanel() {
  const providers = useQuery({
    queryKey: ["ai-providers"],
    queryFn: api.aiListProviders,
    staleTime: 30_000,
  })
  const routes = useQuery({
    queryKey: ["ai-routes"],
    queryFn: api.aiListRoutes,
    staleTime: 30_000,
  })
  const allModels = useQuery({
    queryKey: ["ai-models"],
    queryFn: () => api.aiListModels(),
    staleTime: 30_000,
  })
  // Bundled recommended system prompts keyed by task id. Empty result
  // (e.g. no caption-shaped tasks recommended) is fine — buttons hide.
  const recommended = useQuery({
    queryKey: ["ai-recommended-prompts"],
    queryFn: api.aiListRecommendedPrompts,
    staleTime: 60 * 60 * 1000,
  })

  const providerList = providers.data?.providers ?? []
  const routeMap = useMemo(() => {
    const m = new Map<string, AIRouteRecord>()
    for (const r of routes.data?.routes ?? []) m.set(r.taskId, r)
    return m
  }, [routes.data])

  return (
    <div className="space-y-3">
      <div className="rounded-[4px] border border-border/60 bg-muted/20 px-4 py-2.5 text-[12px] text-muted-foreground/85">
        每项 LoraHub 功能对应一个任务路由。未单独配置的任务使用「默认」兜底路由。
      </div>
      {AI_TASK_IDS.map((taskId) => (
        <RouteRow
          key={taskId}
          taskId={taskId}
          route={routeMap.get(taskId) ?? null}
          providers={providerList}
          allModels={allModels.data?.models ?? []}
          recommendedPrompt={recommended.data?.prompts?.[taskId]}
        />
      ))}
    </div>
  )
}

function RouteRow({
  taskId,
  route,
  providers,
  allModels,
  recommendedPrompt,
}: {
  taskId: AITaskId
  route: AIRouteRecord | null
  providers: AIProviderRecord[]
  allModels: AIModelRecord[]
  recommendedPrompt?: string
}) {
  const qc = useQueryClient()
  const [providerId, setProviderId] = useState(route?.providerId ?? "")
  const [modelId, setModelId] = useState(route?.modelId ?? "")
  const [systemPrompt, setSystemPrompt] = useState(route?.systemPrompt ?? "")
  const [enabled, setEnabled] = useState(route?.enabled ?? true)
  const [advanced, setAdvanced] = useState(false)
  const [temperature, setTemperature] = useState<string>(
    route?.temperature?.toString() ?? "",
  )
  const [maxOutputTokens, setMaxOutputTokens] = useState<string>(
    route?.maxOutputTokens?.toString() ?? "",
  )
  const [reasoningEffort, setReasoningEffort] = useState<string>(
    route?.reasoningEffort ?? "",
  )

  useEffect(() => {
    setProviderId(route?.providerId ?? "")
    setModelId(route?.modelId ?? "")
    setSystemPrompt(route?.systemPrompt ?? "")
    setEnabled(route?.enabled ?? true)
    setTemperature(route?.temperature?.toString() ?? "")
    setMaxOutputTokens(route?.maxOutputTokens?.toString() ?? "")
    setReasoningEffort(route?.reasoningEffort ?? "")
  }, [route?.taskId, route?.updatedAt, route])

  const save = useMutation({
    mutationFn: () =>
      api.aiSaveRoute({
        taskId,
        providerId: providerId || null,
        modelId: modelId || null,
        systemPrompt,
        temperature: temperature.trim() ? Number(temperature) : null,
        maxOutputTokens: maxOutputTokens.trim() ? Number(maxOutputTokens) : null,
        reasoningEffort: (reasoningEffort.trim() || null) as
          | "low"
          | "medium"
          | "high"
          | null,
        stopSequences: route?.stopSequences ?? [],
        extraBodyJson: route?.extraBodyJson ?? "",
        enabled,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ai-routes"] })
    },
  })

  const modelsForProvider = useMemo(
    () =>
      providerId
        ? allModels.filter((m) => m.providerId === providerId && m.enabled)
        : [],
    [allModels, providerId],
  )

  return (
    <Card className="rounded-[6px] border-border/70">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="text-base">{TASK_LABELS[taskId]}</CardTitle>
            <CardDescription className="text-[11px]">
              {TASK_DESCRIPTIONS[taskId]} · <code className="font-mono">{taskId}</code>
            </CardDescription>
          </div>
          <div className="flex items-center gap-1.5">
            <Switch
              checked={enabled}
              onCheckedChange={(v) => setEnabled(v)}
            />
            <span className="text-[11px] text-muted-foreground">启用</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <Field label="服务商">
            <Select
              value={providerId}
              onValueChange={(v) => {
                setProviderId(v ?? "")
                setModelId("")
              }}
            >
              <SelectTrigger className="h-8 text-[12px]">
                <SelectValue placeholder="(继承默认)" />
              </SelectTrigger>
              <SelectContent>
                {providers.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="模型">
            <Select
              value={modelId}
              onValueChange={(v) => setModelId(v ?? "")}
              disabled={!providerId}
            >
              <SelectTrigger className="h-8 text-[12px] font-mono">
                <SelectValue placeholder={providerId ? "选择" : "先选服务商"} />
              </SelectTrigger>
              <SelectContent>
                {modelsForProvider.map((m) => (
                  <SelectItem key={m.id} value={m.modelId}>
                    {m.modelId}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        </div>
        <Field label="System Prompt">
          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={2}
            className="font-mono text-[12px] w-full rounded-[3px] border border-input bg-background/76 px-2 py-1.5"
            placeholder="可选 — 会作为 system 消息加在用户 prompt 之前"
          />
          {recommendedPrompt && recommendedPrompt !== systemPrompt && (
            <button
              type="button"
              onClick={() => setSystemPrompt(recommendedPrompt)}
              className="text-[11px] text-primary hover:underline mt-1"
              title="将 Anima 推荐 caption 模板填入此字段（点保存才会持久化）"
            >
              使用 Anima 推荐 prompt
            </button>
          )}
        </Field>
        <button
          type="button"
          onClick={() => setAdvanced((v) => !v)}
          className="text-[11px] text-muted-foreground hover:text-foreground"
        >
          {advanced ? "收起" : "展开"}采样参数
        </button>
        {advanced && (
          <div className="grid grid-cols-3 gap-2">
            <Field label="temperature">
              <Input
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                placeholder="0.2"
                className="h-8 text-[12px]"
              />
            </Field>
            <Field label="max_output_tokens">
              <Input
                value={maxOutputTokens}
                onChange={(e) => setMaxOutputTokens(e.target.value)}
                placeholder="2048"
                className="h-8 text-[12px]"
              />
            </Field>
            <Field label="reasoning_effort">
              <Select
                value={reasoningEffort}
                onValueChange={(v) => setReasoningEffort(v ?? "")}
              >
                <SelectTrigger className="h-8 text-[12px]">
                  <SelectValue placeholder="(默认)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">(默认)</SelectItem>
                  {REASONING_EFFORTS.map((e) => (
                    <SelectItem key={e} value={e}>
                      {e}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
        )}
      </CardContent>
      <div className="px-4 pb-3 flex items-center justify-end gap-2">
        {save.isError && (
          <span className="text-[11px] font-mono text-destructive">
            {(save.error as Error).message}
          </span>
        )}
        <Button
          size="sm"
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="h-7 text-[11px]"
        >
          {save.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Save className="size-3" />
          )}
          保存
        </Button>
      </div>
    </Card>
  )
}
