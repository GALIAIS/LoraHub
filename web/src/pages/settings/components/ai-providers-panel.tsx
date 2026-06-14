import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bot,
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  Plus,
  Save,
  Search,
  Trash2,
  XCircle,
} from "lucide-react"
import {
  api,
  type AIConnectionTestResult,
  type AIKeySelectionMode,
  type AIProviderDraft,
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
import {
  Field,
  KeyRuntimeBadge,
  makeDraft,
} from "./ai-provider-form-support"

const SELECTION_MODES: { value: AIKeySelectionMode; label: string }[] = [
  { value: "round_robin", label: "轮询" },
  { value: "random", label: "随机" },
]

export function ProvidersPanel() {
  const providers = useQuery({
    queryKey: ["ai-providers"],
    queryFn: api.aiListProviders,
    staleTime: 30_000,
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
            尚未配置服务商。点「新增」开始，可任意添加 OpenAI 兼容端点。
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
            选一个服务商开始编辑，或点上方「新增」添加。
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
