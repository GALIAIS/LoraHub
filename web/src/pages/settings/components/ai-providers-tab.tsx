import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bot,
  CheckCircle2,
  Eye,
  EyeOff,
  ExternalLink,
  Loader2,
  Save,
  Trash2,
  XCircle,
  Zap,
} from "lucide-react"
import {
  api,
  type AIProviderEntry,
  type AITestResult,
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
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

/**
 * Settings → AI 服务商
 *
 * One card per provider; each card lets the user paste a key, optionally
 * override base_url + default model, save, test, delete. All keys live
 * in the dedicated `runs/ai_credentials.sqlite` so they don't pollute
 * the main settings.json.
 */
export function AIProvidersTab() {
  const providers = useQuery({
    queryKey: ["ai-providers"],
    queryFn: api.aiListProviders,
  })

  if (providers.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" /> 正在加载服务商目录…
      </div>
    )
  }
  if (providers.isError) {
    return (
      <div className="text-xs text-destructive font-mono">
        {(providers.error as Error).message}
      </div>
    )
  }
  const list = providers.data?.providers ?? []
  const configured = list.filter((p) => p.configured).length

  return (
    <div className="space-y-4">
      <div className="rounded-[4px] border border-border/60 bg-muted/20 px-4 py-3 text-sm flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Bot className="size-4 text-muted-foreground" />
          <span>
            已配置{" "}
            <code className="font-mono font-semibold text-foreground">
              {configured}
            </code>{" "}
            / {list.length} 家服务商
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground/85">
          API Key 单独存于{" "}
          <code className="font-mono">runs/ai_credentials.sqlite</code>,文件权限 600
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {list.map((p) => (
          <ProviderCard key={p.id} entry={p} />
        ))}
      </div>
    </div>
  )
}

function ProviderCard({ entry }: { entry: AIProviderEntry }) {
  const qc = useQueryClient()
  const [apiKey, setApiKey] = useState("")
  const [baseUrl, setBaseUrl] = useState(entry.current_base_url ?? "")
  const [defaultModel, setDefaultModel] = useState(
    entry.current_default_model ?? entry.default_model ?? "",
  )
  const [enabled, setEnabled] = useState(entry.enabled || !entry.configured)
  const [showKey, setShowKey] = useState(false)
  const [testResult, setTestResult] = useState<AITestResult | null>(null)

  // Re-sync local form when the upstream entry changes (e.g. after a save).
  useEffect(() => {
    setBaseUrl(entry.current_base_url ?? "")
    setDefaultModel(entry.current_default_model ?? entry.default_model ?? "")
    setEnabled(entry.enabled || !entry.configured)
  }, [entry.id, entry.current_base_url, entry.current_default_model, entry.default_model, entry.configured, entry.enabled])

  const save = useMutation({
    mutationFn: () =>
      api.aiUpsertCredential({
        provider: entry.id,
        api_key: apiKey.trim() || null,
        base_url: baseUrl.trim() || null,
        default_model: defaultModel.trim() || null,
        enabled,
      }),
    onSuccess: () => {
      setApiKey("")
      setTestResult(null)
      qc.invalidateQueries({ queryKey: ["ai-providers"] })
    },
  })

  const remove = useMutation({
    mutationFn: () => api.aiDeleteCredential(entry.id),
    onSuccess: () => {
      setApiKey("")
      setTestResult(null)
      qc.invalidateQueries({ queryKey: ["ai-providers"] })
    },
  })

  const test = useMutation({
    mutationFn: () =>
      api.aiTestProvider({
        provider: entry.id,
        api_key: apiKey.trim() || null,
        base_url: baseUrl.trim() || null,
        model: defaultModel.trim() || null,
      }),
    onSuccess: (r) => setTestResult(r),
  })

  const visionModelCount = useMemo(
    () => entry.models.filter((m) => m.vision).length,
    [entry.models],
  )

  return (
    <Card
      className={cn(
        "rounded-[6px] border-border/70 shadow-[var(--panel-shadow)] flex flex-col",
        entry.configured && entry.enabled && "border-primary/40",
      )}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="text-base flex items-center gap-2">
              {entry.name}
              {entry.configured && entry.enabled && (
                <Badge variant="secondary" className="rounded-[2px] text-[10px]">
                  已启用
                </Badge>
              )}
              {entry.configured && !entry.enabled && (
                <Badge variant="outline" className="rounded-[2px] text-[10px]">
                  已禁用
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="text-[11px]">
              {entry.models.length} 个模型
              {visionModelCount > 0 && (
                <>
                  {" · "}
                  <span className="text-foreground">{visionModelCount}</span>{" "}
                  个支持视觉
                </>
              )}
            </CardDescription>
          </div>
          {entry.docs_url && (
            <a
              href={entry.docs_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] text-muted-foreground hover:text-foreground inline-flex items-center gap-1 shrink-0"
            >
              文档 <ExternalLink className="size-3" />
            </a>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-2.5 flex-1">
        <p className="text-[11px] text-muted-foreground/85 leading-relaxed">
          {entry.auth_help}
        </p>
        <div className="space-y-1.5">
          <Label className="text-[11px]">API Key</Label>
          <div className="relative">
            <Input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                entry.configured
                  ? "已保存(留空保持不变)"
                  : "粘贴你的 API Key"
              }
              className="font-mono text-[12px] pr-8"
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              {showKey ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
            </button>
          </div>
        </div>

        {entry.custom_base_url && (
          <div className="space-y-1.5">
            <Label className="text-[11px]">Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={entry.default_base_url}
              className="font-mono text-[12px]"
            />
          </div>
        )}

        <div className="space-y-1.5">
          <Label className="text-[11px]">默认模型</Label>
          {entry.models.length > 0 && !entry.custom_base_url ? (
            <Select
              value={defaultModel}
              onValueChange={(v) => setDefaultModel(v ?? "")}
            >
              <SelectTrigger className="h-8 text-[12px] font-mono">
                <SelectValue placeholder="选择模型" />
              </SelectTrigger>
              <SelectContent>
                {entry.models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    <span className="font-mono text-[12px]">{m.id}</span>
                    <span className="ml-2 text-[11px] text-muted-foreground">
                      {m.label}
                      {m.vision && " · 视觉"}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              placeholder={entry.default_model ?? "model-id"}
              className="font-mono text-[12px]"
            />
          )}
        </div>

        <div className="flex items-center justify-between gap-2 pt-1">
          <Label className="text-[11px] flex items-center gap-2">
            启用
            <Switch
              checked={enabled}
              onCheckedChange={(v) => setEnabled(v)}
            />
          </Label>
        </div>

        {testResult && (
          <div
            className={cn(
              "rounded-[3px] border px-2 py-1.5 text-[11px] font-mono break-all",
              testResult.ok
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : "border-destructive/40 bg-destructive/5 text-destructive",
            )}
          >
            {testResult.ok ? (
              <span className="flex items-center gap-1">
                <CheckCircle2 className="size-3.5 shrink-0" />
                连接成功 · {testResult.model ?? "-"}
              </span>
            ) : (
              <span className="flex items-start gap-1">
                <XCircle className="size-3.5 shrink-0 mt-0.5" />
                <span>{testResult.error ?? "请求失败"}</span>
              </span>
            )}
          </div>
        )}
      </CardContent>
      <div className="px-4 pb-3 pt-0 flex items-center justify-between gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={
            test.isPending ||
            (!apiKey.trim() && !entry.configured) ||
            (entry.custom_base_url && !baseUrl.trim() && !entry.current_base_url)
          }
          onClick={() => test.mutate()}
          className="h-7 text-[11px]"
        >
          {test.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Zap className="size-3" />
          )}
          测试
        </Button>
        <div className="flex items-center gap-2">
          {entry.configured && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
              className="h-7 text-[11px] text-destructive hover:text-destructive"
            >
              <Trash2 className="size-3" /> 删除
            </Button>
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
      </div>
      {save.isError && (
        <div className="px-4 pb-2 -mt-2 text-[11px] font-mono text-destructive">
          {(save.error as Error).message}
        </div>
      )}
    </Card>
  )
}
