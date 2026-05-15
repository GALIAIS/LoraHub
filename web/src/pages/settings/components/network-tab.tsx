import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Save, RotateCcw, Globe2, Github, Cloud } from "lucide-react"
import { api, type SettingsState } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"

const GITHUB_PROXY_PRESETS = [
  { label: "直连（不使用代理）", value: "" },
  { label: "gh-proxy.org", value: "https://gh-proxy.org" },
  { label: "hk.gh-proxy.org（香港节点）", value: "https://hk.gh-proxy.org" },
  { label: "cdn.gh-proxy.org（CDN 节点）", value: "https://cdn.gh-proxy.org" },
  { label: "edgeone.gh-proxy.org（EdgeOne 节点）", value: "https://edgeone.gh-proxy.org" },
] as const

const HF_PRESETS = [
  { label: "huggingface.co（官方）", value: "" },
  { label: "hf-mirror.com（国内镜像）", value: "https://hf-mirror.com" },
] as const

type Draft = {
  github_proxy: string
  huggingface_endpoint: string
  modelscope_enabled: boolean
  modelscope_token: string
}

function buildDraft(s: SettingsState): Draft {
  return {
    github_proxy: s.github_proxy ?? "",
    huggingface_endpoint: s.huggingface_endpoint ?? "",
    modelscope_enabled: s.modelscope_enabled,
    modelscope_token: s.modelscope_token ?? "",
  }
}

/**
 * Network acceleration: GitHub proxy + HuggingFace mirror + ModelScope.
 *
 * The dirty state is local to this tab so saves don't collide with edits in
 * sibling tabs.
 */
export function NetworkTab() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const [draft, setDraft] = useState<Draft | null>(null)

  useEffect(() => {
    if (settingsQuery.data) setDraft(buildDraft(settingsQuery.data.settings))
  }, [settingsQuery.data])

  const update = useMutation({
    mutationFn: (patch: Partial<SettingsState>) => api.updateSettings(patch),
    onSuccess: (data) => {
      qc.setQueryData(["settings"], data)
      qc.invalidateQueries({ queryKey: ["health"] })
      qc.invalidateQueries({ queryKey: ["backends"] })
      setDraft(buildDraft(data.settings))
    },
  })

  if (!draft || !settingsQuery.data) {
    return <div className="text-sm text-muted-foreground">正在加载设置…</div>
  }

  const saved = settingsQuery.data.settings
  const dirty =
    draft.github_proxy !== (saved.github_proxy ?? "") ||
    draft.huggingface_endpoint !== (saved.huggingface_endpoint ?? "") ||
    draft.modelscope_enabled !== saved.modelscope_enabled ||
    draft.modelscope_token !== (saved.modelscope_token ?? "")

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Github className="size-4 text-muted-foreground" />
            GitHub 代理
          </CardTitle>
          <CardDescription>
            国内访问 GitHub 不畅时，可在此选用代理。安装 kohya / diffusion-pipe 时
            <code className="text-foreground"> git clone </code>
            将自动改写为 <code className="text-foreground">代理/原 URL</code>。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {GITHUB_PROXY_PRESETS.map((p) => (
              <Button
                key={p.label}
                type="button"
                size="sm"
                variant={draft.github_proxy === p.value ? "default" : "outline"}
                onClick={() => setDraft({ ...draft, github_proxy: p.value })}
              >
                {p.label}
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">自定义</Label>
            <Input
              value={draft.github_proxy}
              placeholder="https://gh-proxy.org（留空表示直连）"
              onChange={(e) => setDraft({ ...draft, github_proxy: e.target.value })}
              className="font-mono"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Globe2 className="size-4 text-muted-foreground" />
            HuggingFace 镜像
          </CardTitle>
          <CardDescription>
            训练 / 标注 / 模型下载子进程启动时会注入
            <code className="text-foreground"> HF_ENDPOINT </code>
            环境变量。已设置同名环境变量时本字段不生效。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {HF_PRESETS.map((p) => (
              <Button
                key={p.label}
                type="button"
                size="sm"
                variant={
                  draft.huggingface_endpoint === p.value ? "default" : "outline"
                }
                onClick={() =>
                  setDraft({ ...draft, huggingface_endpoint: p.value })
                }
              >
                {p.label}
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">自定义</Label>
            <Input
              value={draft.huggingface_endpoint}
              placeholder="https://hf-mirror.com（留空表示官方站）"
              onChange={(e) =>
                setDraft({ ...draft, huggingface_endpoint: e.target.value })
              }
              className="font-mono"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cloud className="size-4 text-muted-foreground" />
            ModelScope（魔搭）
          </CardTitle>
          <CardDescription>
            国内更稳的模型源。可在「模型下载」标签直接拉取。私有模型需填写访问令牌。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">优先 ModelScope</Label>
            <Switch
              checked={draft.modelscope_enabled}
              onCheckedChange={(v) =>
                setDraft({ ...draft, modelscope_enabled: v })
              }
            />
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-start">
            <Label className="text-xs pt-2">访问令牌</Label>
            <div className="space-y-1">
              <Input
                type="password"
                value={draft.modelscope_token}
                placeholder="（可选，仅私有模型需要）"
                onChange={(e) =>
                  setDraft({ ...draft, modelscope_token: e.target.value })
                }
                className="font-mono"
              />
              <p className="text-[11px] text-muted-foreground/80">
                在 <code>https://modelscope.cn</code> 个人中心生成。保存后用作
                Bearer 令牌注入下载请求与子进程
                <code className="text-foreground"> MODELSCOPE_API_TOKEN </code>
                环境变量。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center gap-3 sticky bottom-4 bg-background/80 backdrop-blur rounded-[4px] border border-border/60 px-4 py-3 shadow-[var(--panel-shadow)]">
        <Button
          size="sm"
          disabled={!dirty || update.isPending}
          onClick={() =>
            update.mutate({
              github_proxy: draft.github_proxy || null,
              huggingface_endpoint: draft.huggingface_endpoint || null,
              modelscope_enabled: draft.modelscope_enabled,
              modelscope_token: draft.modelscope_token || null,
            })
          }
        >
          <Save className="size-3" />
          {update.isPending ? "保存中…" : "保存"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!dirty || update.isPending}
          onClick={() => settingsQuery.data && setDraft(buildDraft(settingsQuery.data.settings))}
        >
          <RotateCcw className="size-3" />
          重置
        </Button>
        {update.isError && (
          <span className="text-xs text-destructive font-mono">
            {(update.error as Error).message}
          </span>
        )}
        {update.isSuccess && !dirty && (
          <span className="text-xs text-emerald-600 dark:text-emerald-400">
            已保存。
          </span>
        )}
      </div>
    </div>
  )
}
