import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bot,
  CheckCircle2,
  Eye,
  EyeOff,
  Layers,
  Loader2,
  Plus,
  Save,
  Search,
  Settings2,
  Trash2,
  XCircle,
} from "lucide-react"
import {
  api,
  AI_TASK_IDS,
  type AIConnectionTestResult,
  type AIKeySelectionMode,
  type AIModelRecord,
  type AIProviderDraft,
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

const SELECTION_MODES: { value: AIKeySelectionMode; label: string }[] = [
  { value: "round_robin", label: "轮询" },
  { value: "random", label: "随机" },
]

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

// --------------------------------------------------------------------------- //
// Providers panel: list + edit form
// --------------------------------------------------------------------------- //

function ProvidersPanel() {
  const providers = useQuery({
    queryKey: ["ai-providers"],
    queryFn: api.aiListProviders,
  })
  const [editing, setEditing] = useState<string | "new" | null>(null)

  const list = providers.data?.providers ?? []
  const editingProvider = useMemo(() => {
    if (editing === "new") return null
    if (editing == null) return null
    return list.find((p) => p.id === editing) ?? null
  }, [editing, list])

  return (
    <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="text-sm font-medium flex items-center gap-2">
            <Bot className="size-3.5 text-muted-foreground" />
            服务商
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-[11px]"
            onClick={() => setEditing("new")}
          >
            <Plus className="size-3" /> 新增
          </Button>
        </div>
        {providers.isLoading && (
          <div className="text-[11px] text-muted-foreground flex items-center gap-1.5">
            <Loader2 className="size-3 animate-spin" /> 加载中…
          </div>
        )}
        {providers.isError && (
          <div className="text-[11px] text-destructive font-mono">
            {(providers.error as Error).message}
          </div>
        )}
        {list.length === 0 && !providers.isLoading && (
          <div className="rounded-[4px] border border-dashed border-border/60 px-3 py-4 text-[11px] text-muted-foreground/85">
            尚未配置服务商。点「新增」开始,可任意添加 OpenAI 兼容端点。
          </div>
        )}
        <div className="space-y-1">
          {list.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setEditing(p.id)}
              className={cn(
                "w-full text-left rounded-[3px] border px-2.5 py-1.5 text-[12px] transition-colors",
                editing === p.id
                  ? "border-primary bg-primary/5"
                  : "border-border/60 hover:bg-muted/40",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium truncate">{p.name || "(未命名)"}</span>
                {p.enabled ? (
                  <Badge variant="secondary" className="text-[9px] rounded-[2px]">启用</Badge>
                ) : (
                  <Badge variant="outline" className="text-[9px] rounded-[2px]">禁用</Badge>
                )}
              </div>
              <div className="text-[10px] text-muted-foreground/85 mt-0.5 flex items-center gap-2">
                <span className="font-mono truncate">{p.baseUrl || "(无 base URL)"}</span>
                <span className="shrink-0">{p.apiKeyCount} keys</span>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="min-w-0">
        {editing == null ? (
          <div className="rounded-[6px] border border-dashed border-border/60 px-6 py-12 text-center text-sm text-muted-foreground">
            选一个服务商开始编辑,或点上方「新增」添加。
          </div>
        ) : (
          <ProviderForm
            key={editing}
            existing={editingProvider}
            isNew={editing === "new"}
            onDeleted={() => setEditing(null)}
            onSaved={(id) => setEditing(id)}
          />
        )}
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Provider form
// --------------------------------------------------------------------------- //

interface KeyDraftLocal {
  id: string | null
  preview: string
  value: string
  // Runtime is only populated for already-saved keys
  requestCount?: number
  successCount?: number
  failureCount?: number
  cooldownUntil?: string | null
  lastError?: string | null
}

function makeDraft(provider: AIProviderRecord | null): {
  name: string
  baseUrl: string
  organization: string
  project: string
  enabled: boolean
  selectionMode: AIKeySelectionMode
  keys: KeyDraftLocal[]
  headersJson: string
} {
  return {
    name: provider?.name ?? "",
    baseUrl: provider?.baseUrl ?? "",
    organization: provider?.organization ?? "",
    project: provider?.project ?? "",
    enabled: provider?.enabled ?? true,
    selectionMode: provider?.apiKeySelectionMode ?? "round_robin",
    keys: provider?.apiKeys.map((k) => ({
      id: k.id,
      preview: k.preview,
      value: "",
      requestCount: k.runtime.requestCount,
      successCount: k.runtime.successCount,
      failureCount: k.runtime.failureCount,
      cooldownUntil: k.runtime.cooldownUntil,
      lastError: k.runtime.lastError,
    })) ?? [],
    headersJson:
      provider && Object.keys(provider.headers).length > 0
        ? JSON.stringify(provider.headers, null, 2)
        : "",
  }
}

function ProviderForm({
  existing,
  isNew,
  onSaved,
  onDeleted,
}: {
  existing: AIProviderRecord | null
  isNew: boolean
  onSaved: (id: string) => void
  onDeleted: () => void
}) {
  const qc = useQueryClient()
  const [draft, setDraft] = useState(() => makeDraft(existing))
  const [showKeyById, setShowKeyById] = useState<Record<string, boolean>>({})
  const [test, setTest] = useState<AIConnectionTestResult | null>(null)

  useEffect(() => {
    setDraft(makeDraft(existing))
    setTest(null)
  }, [existing?.id, existing?.updatedAt, existing])

  const headersError = (() => {
    if (!draft.headersJson.trim()) return null
    try {
      const parsed = JSON.parse(draft.headersJson)
      if (typeof parsed !== "object" || parsed == null || Array.isArray(parsed)) {
        return "headers 必须是 JSON 对象"
      }
      return null
    } catch (e) {
      return (e as Error).message
    }
  })()

  const save = useMutation({
    mutationFn: async (): Promise<{ provider: AIProviderRecord }> => {
      const headers = draft.headersJson.trim() ? JSON.parse(draft.headersJson) : {}
      const payload: AIProviderDraft = {
        id: existing?.id,
        name: draft.name.trim(),
        kind: "openai-compatible",
        baseUrl: draft.baseUrl.trim(),
        organization: draft.organization.trim(),
        project: draft.project.trim(),
        headers,
        enabled: draft.enabled,
        apiKeySelectionMode: draft.selectionMode,
        apiKeys: draft.keys.map((k) => ({
          id: k.id,
          value: k.value || undefined,
        })),
      }
      return api.aiSaveProvider(payload)
    },
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["ai-providers"] })
      qc.invalidateQueries({ queryKey: ["ai-models"] })
      onSaved(r.provider.id)
    },
  })

  const remove = useMutation({
    mutationFn: () =>
      existing ? api.aiDeleteProvider(existing.id) : Promise.resolve({ ok: false, providerId: "" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ai-providers"] })
      onDeleted()
    },
  })

  const discover = useMutation({
    mutationFn: () =>
      existing ? api.aiDiscoverModels(existing.id) : Promise.resolve({ models: [] }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ai-models"] })
    },
  })

  const testConn = useMutation({
    mutationFn: () =>
      existing
        ? api.aiTestConnection({ providerId: existing.id })
        : Promise.resolve(null as unknown as AIConnectionTestResult),
    onSuccess: (r) => setTest(r),
  })

  const canSave =
    draft.name.trim() &&
    draft.baseUrl.trim() &&
    headersError == null &&
    !save.isPending

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">
          {isNew ? "新建服务商" : draft.name || "(未命名)"}
        </CardTitle>
        <CardDescription className="text-[11px]">
          所有服务商使用 OpenAI 兼容协议 (Bearer + /v1/chat/completions)。
          可填任意自部署 / 公有云 / 中转 base URL。
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Field label="名称">
          <Input
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            placeholder="例: DeepSeek / 我的 Ollama"
          />
        </Field>
        <Field
          label="Base URL"
          hint="可填带或不带 /v1 后缀(例: https://api.deepseek.com 或 https://api.deepseek.com/v1)"
        >
          <Input
            value={draft.baseUrl}
            onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
            placeholder="https://api.deepseek.com"
            className="font-mono text-[12px]"
          />
        </Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Organization (可选)">
            <Input
              value={draft.organization}
              onChange={(e) => setDraft({ ...draft, organization: e.target.value })}
            />
          </Field>
          <Field label="Project (可选)">
            <Input
              value={draft.project}
              onChange={(e) => setDraft({ ...draft, project: e.target.value })}
            />
          </Field>
        </div>
        <Field
          label="自定义 Headers (JSON)"
          hint='形如 { "X-Foo": "bar" }; 留空表示无自定义 header'
        >
          <textarea
            value={draft.headersJson}
            onChange={(e) => setDraft({ ...draft, headersJson: e.target.value })}
            placeholder='{ "X-Tenant": "abc" }'
            className="font-mono text-[12px] w-full min-h-[60px] rounded-[3px] border border-input bg-background/76 px-2 py-1.5 text-foreground"
          />
          {headersError && (
            <div className="text-[11px] font-mono text-destructive">{headersError}</div>
          )}
        </Field>

        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Switch
              checked={draft.enabled}
              onCheckedChange={(v) => setDraft({ ...draft, enabled: v })}
            />
            <Label className="text-[12px]">启用此服务商</Label>
          </div>
          <Field label="多 key 调度">
            <Select
              value={draft.selectionMode}
              onValueChange={(v) =>
                setDraft({ ...draft, selectionMode: (v ?? "round_robin") as AIKeySelectionMode })
              }
            >
              <SelectTrigger className="h-8 text-[12px] w-[7rem]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SELECTION_MODES.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <Label className="text-[12px]">API Keys</Label>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-[11px]"
              onClick={() =>
                setDraft({
                  ...draft,
                  keys: [...draft.keys, { id: null, preview: "", value: "" }],
                })
              }
            >
              <Plus className="size-3" /> 添加 key
            </Button>
          </div>
          {draft.keys.length === 0 ? (
            <div className="rounded-[3px] border border-dashed border-border/60 px-3 py-3 text-[11px] text-muted-foreground/85">
              至少添加一个 API key 才能使用这家服务商。
            </div>
          ) : (
            <div className="space-y-2">
              {draft.keys.map((k, idx) => {
                const showId = k.id ?? `new-${idx}`
                const visible = showKeyById[showId] ?? false
                return (
                  <div
                    key={showId}
                    className="rounded-[3px] border border-border/60 p-2 space-y-1.5"
                  >
                    <div className="flex items-center gap-2">
                      <div className="relative flex-1">
                        <Input
                          type={visible ? "text" : "password"}
                          value={k.value}
                          onChange={(e) => {
                            const next = [...draft.keys]
                            next[idx] = { ...next[idx], value: e.target.value }
                            setDraft({ ...draft, keys: next })
                          }}
                          placeholder={k.id ? `已保存 (${k.preview})` : "粘贴 API key"}
                          className="font-mono text-[11px] pr-8"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setShowKeyById((m) => ({ ...m, [showId]: !visible }))
                          }
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        >
                          {visible ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                        </button>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-1.5 text-destructive hover:text-destructive"
                        onClick={() =>
                          setDraft({
                            ...draft,
                            keys: draft.keys.filter((_, i) => i !== idx),
                          })
                        }
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    </div>
                    {k.id && (
                      <KeyRuntimeBadge k={k} />
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {test && (
          <div
            className={cn(
              "rounded-[3px] border px-2 py-1.5 text-[11px] font-mono break-all",
              test.ok
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : "border-destructive/40 bg-destructive/5 text-destructive",
            )}
          >
            {test.ok ? (
              <span className="flex items-center gap-1">
                <CheckCircle2 className="size-3.5 shrink-0" />
                连接成功 · 发现 {test.modelCount} 个模型
              </span>
            ) : (
              <span className="flex items-start gap-1">
                <XCircle className="size-3.5 shrink-0 mt-0.5" />
                <span>{test.error ?? "请求失败"}</span>
              </span>
            )}
          </div>
        )}
      </CardContent>
      <div className="px-4 pb-3 pt-0 flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          {!isNew && existing && (
            <>
              <Button
                size="sm"
                variant="outline"
                onClick={() => testConn.mutate()}
                disabled={testConn.isPending}
                className="h-7 text-[11px]"
              >
                {testConn.isPending ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <CheckCircle2 className="size-3" />
                )}
                测试
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => discover.mutate()}
                disabled={discover.isPending}
                className="h-7 text-[11px]"
                title="从 /v1/models 拉取并刷新此服务商可用的模型列表"
              >
                {discover.isPending ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Search className="size-3" />
                )}
                发现模型
              </Button>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!isNew && existing && (
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-[11px] text-destructive hover:text-destructive"
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
            >
              <Trash2 className="size-3" /> 删除
            </Button>
          )}
          <Button
            size="sm"
            onClick={() => save.mutate()}
            disabled={!canSave}
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
      </div>
      {(save.isError || remove.isError || discover.isError || testConn.isError) && (
        <div className="px-4 pb-2 -mt-2 text-[11px] font-mono text-destructive">
          {String(
            (save.error || remove.error || discover.error || testConn.error) as Error,
          )}
        </div>
      )}
    </Card>
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

function KeyRuntimeBadge({ k }: { k: KeyDraftLocal }) {
  const onCooldown = !!k.cooldownUntil && new Date(k.cooldownUntil) > new Date()
  return (
    <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
      <span>请求 {k.requestCount ?? 0}</span>
      <span className="text-emerald-600 dark:text-emerald-400">
        ✓{k.successCount ?? 0}
      </span>
      <span className="text-destructive">
        ✗{k.failureCount ?? 0}
      </span>
      {onCooldown && (
        <Badge variant="destructive" className="text-[9px] rounded-[2px]">
          冷却中
        </Badge>
      )}
      {k.lastError && (
        <span className="truncate text-destructive/85" title={k.lastError}>
          · {k.lastError}
        </span>
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
  })
  const [providerFilter, setProviderFilter] = useState<string>("all")
  const models = useQuery({
    queryKey: ["ai-models", providerFilter],
    queryFn: () =>
      providerFilter === "all"
        ? api.aiListModels()
        : api.aiListModels(providerFilter),
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
          先到「服务商」面板添加至少一个服务商,再来管理模型。
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
              ? "尚无任何模型。选一个服务商点「发现模型」自动拉取,或手工添加。"
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
              <CardContent className="space-y-1">
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
  })
  const routes = useQuery({
    queryKey: ["ai-routes"],
    queryFn: api.aiListRoutes,
  })
  const allModels = useQuery({
    queryKey: ["ai-models"],
    queryFn: () => api.aiListModels(),
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
