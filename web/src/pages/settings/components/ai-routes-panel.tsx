import { useEffect, useMemo, useState, type ReactNode } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, Save } from "lucide-react"
import {
  api,
  AI_TASK_IDS,
  type AIModelRecord,
  type AIProviderRecord,
  type AIRouteRecord,
  type AITaskId,
} from "@/lib/api"
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

export function RoutesPanel() {
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
