import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Cloud,
  Github,
  Globe2,
  Loader2,
  RotateCcw,
  Save,
  Sparkles,
  Zap,
} from "lucide-react"
import {
  api,
  type MirrorPreset,
  type ProbeResult,
  type SettingsState,
} from "@/lib/api"
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
import { cn } from "@/lib/utils"

type Draft = {
  github_proxy: string
  huggingface_endpoint: string
  huggingface_token: string
  modelscope_enabled: boolean
  modelscope_token: string
  pypi_index_url: string
  torch_index_url: string
  download_proxy: string
  wandb_api_key: string
  wandb_base_url: string
}

function buildDraft(s: SettingsState): Draft {
  return {
    github_proxy: s.github_proxy ?? "",
    huggingface_endpoint: s.huggingface_endpoint ?? "",
    huggingface_token: s.huggingface_token ?? "",
    modelscope_enabled: s.modelscope_enabled,
    modelscope_token: s.modelscope_token ?? "",
    pypi_index_url: s.pypi_index_url ?? "",
    torch_index_url: s.torch_index_url ?? "",
    download_proxy: s.download_proxy ?? "",
    wandb_api_key: s.wandb_api_key ?? "",
    wandb_base_url: s.wandb_base_url ?? "",
  }
}

function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—"
  if (ms < 100) return `${ms.toFixed(0)} ms`
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

function latencyTone(ms: number | null | undefined, ok: boolean): string {
  if (!ok || ms === null || ms === undefined) return "text-destructive"
  if (ms < 200) return "text-emerald-600 dark:text-emerald-400"
  if (ms < 500) return "text-primary"
  if (ms < 1500) return "text-amber-600 dark:text-amber-400"
  return "text-destructive"
}

interface MirrorSelectorProps {
  category: "github_proxy" | "huggingface" | "pypi"
  presets: MirrorPreset[]
  current: string
  onChoose: (value: string) => void
  /** Called when the user clicks "测速并自动选用最快" with a fresh result. */
  onAutoPick?: (result: ProbeResult) => void
}

function MirrorSelector({
  category,
  presets,
  current,
  onChoose,
  onAutoPick,
}: MirrorSelectorProps) {
  const [results, setResults] = useState<ProbeResult[] | null>(null)

  const probe = useMutation({
    mutationFn: () => api.probeMirrors({ category }),
    onSuccess: (rows) => setResults(rows),
  })

  // Auto-pick the fastest reachable mirror once the probe lands.
  useEffect(() => {
    if (!probe.isSuccess || !probe.data) return
    const fastest = probe.data.find((r) => r.ok)
    if (fastest) {
      onChoose(fastest.value)
      onAutoPick?.(fastest)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [probe.isSuccess])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          可选镜像
        </span>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={probe.isPending}
          onClick={() => {
            setResults(null)
            probe.mutate()
          }}
        >
          {probe.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Zap className="size-3" />
          )}
          {probe.isPending ? "测速中…" : "测速并自动选用最快"}
        </Button>
      </div>

      <div className="rounded-[4px] border border-border/60 divide-y divide-border/40 overflow-hidden">
        {presets.map((p) => {
          const r = results?.find((x) => x.value === p.value)
          const fastest =
            results !== null && results.length > 0 && results.find((x) => x.ok)?.value
          const isCurrent = current === p.value
          const isFastest = r && r.value === fastest && r.ok
          return (
            <button
              key={p.label + p.value}
              type="button"
              onClick={() => onChoose(p.value)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 text-xs text-left transition-colors",
                isCurrent
                  ? "bg-primary/10 text-foreground"
                  : "hover:bg-muted/50 text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="flex-1 min-w-0">
                <span className="block font-medium truncate">{p.label}</span>
                {p.value && (
                  <span className="block text-[10px] font-mono text-muted-foreground/70 truncate">
                    {p.value}
                  </span>
                )}
              </span>
              {r && (
                <span
                  className={cn(
                    "text-[11px] font-mono tabular-nums shrink-0",
                    latencyTone(r.latency_ms, r.ok),
                  )}
                  title={r.error ?? undefined}
                >
                  {r.ok ? formatLatency(r.latency_ms) : "不可达"}
                </span>
              )}
              {isFastest && (
                <span className="rounded-[2px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.1em] shrink-0 inline-flex items-center gap-1">
                  <Sparkles className="size-2.5" />
                  最快
                </span>
              )}
              {isCurrent && (
                <span className="rounded-[2px] bg-primary/10 text-primary px-1.5 py-0.5 text-[10px] uppercase tracking-[0.1em] shrink-0">
                  已选
                </span>
              )}
            </button>
          )
        })}
      </div>

      {probe.isError && (
        <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-1.5 text-xs font-mono text-destructive">
          {(probe.error as Error).message}
        </div>
      )}
    </div>
  )
}

/**
 * Network acceleration: GitHub proxy + HuggingFace mirror + ModelScope.
 *
 * Each mirror category supports a "test latency and pick the fastest"
 * action backed by /api/network/probe; the user can also click any preset
 * directly. The dirty state is local to this tab so saves don't collide
 * with edits in sibling tabs.
 */
export function NetworkTab() {
  const qc = useQueryClient()
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 30_000,
  })

  const presetsQuery = useQuery({
    queryKey: ["mirror-presets"],
    queryFn: api.listMirrorPresets,
    staleTime: 60 * 60 * 1000, // 1h — preset list is effectively static
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
    draft.huggingface_token !== (saved.huggingface_token ?? "") ||
    draft.modelscope_enabled !== saved.modelscope_enabled ||
    draft.modelscope_token !== (saved.modelscope_token ?? "") ||
    draft.pypi_index_url !== (saved.pypi_index_url ?? "") ||
    draft.torch_index_url !== (saved.torch_index_url ?? "") ||
    draft.download_proxy !== (saved.download_proxy ?? "") ||
    draft.wandb_api_key !== (saved.wandb_api_key ?? "") ||
    draft.wandb_base_url !== (saved.wandb_base_url ?? "")

  const githubPresets = presetsQuery.data?.github_proxy ?? []
  const hfPresets = presetsQuery.data?.huggingface ?? []
  const pypiPresets = presetsQuery.data?.pypi ?? []

  return (
    <div className="space-y-5">
      <Card>
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
          <MirrorSelector
            category="github_proxy"
            presets={githubPresets}
            current={draft.github_proxy}
            onChoose={(v) => setDraft({ ...draft, github_proxy: v })}
          />
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

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Globe2 className="size-4 text-muted-foreground" />
            HuggingFace 镜像
          </CardTitle>
          <CardDescription>
            内置模型下载会直接使用这里的 endpoint。训练 / 标注子进程启动时会在未设置
            <code className="text-foreground"> HF_ENDPOINT </code> 的情况下自动注入。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <MirrorSelector
            category="huggingface"
            presets={hfPresets}
            current={draft.huggingface_endpoint}
            onChoose={(v) => setDraft({ ...draft, huggingface_endpoint: v })}
          />
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">自定义</Label>
            <Input
              value={draft.huggingface_endpoint}
              placeholder="https://hf-mirror.com(留空表示官方站)"
              onChange={(e) =>
                setDraft({ ...draft, huggingface_endpoint: e.target.value })
              }
              className="font-mono"
            />
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-start">
            <Label className="text-xs pt-2">访问令牌</Label>
            <div className="space-y-1">
              <Input
                type="password"
                value={draft.huggingface_token}
                placeholder="hf_xxx (受限仓库需要,如 black-forest-labs/FLUX)"
                onChange={(e) =>
                  setDraft({ ...draft, huggingface_token: e.target.value })
                }
                className="font-mono"
              />
              <p className="text-[11px] text-muted-foreground/85 leading-relaxed">
                作为 <code className="text-foreground">HF_TOKEN</code> 注入下载与训练子进程,
                同时用于 hub API 鉴权。可在 huggingface.co/settings/tokens 创建。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
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

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cloud className="size-4 text-muted-foreground" />
            PyPI 镜像（pip 依赖源）
          </CardTitle>
          <CardDescription>
            后端 venv 安装 Python 依赖（kohya / diffusion-pipe requirements、xformers
            等普通包）时使用的 PyPI 索引。已显式指定的 wheel 源
            （如 <code>download.pytorch.org</code>）不会被改写。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <MirrorSelector
            category="pypi"
            presets={pypiPresets}
            current={draft.pypi_index_url}
            onChoose={(v) => setDraft({ ...draft, pypi_index_url: v })}
          />
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">自定义</Label>
            <Input
              value={draft.pypi_index_url}
              placeholder="https://pypi.tuna.tsinghua.edu.cn/simple（留空使用 pypi.org）"
              onChange={(e) =>
                setDraft({ ...draft, pypi_index_url: e.target.value })
              }
              className="font-mono"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cloud className="size-4 text-muted-foreground" />
            PyTorch 镜像（torch / torchvision / xformers）
          </CardTitle>
          <CardDescription>
            kohya 与 diffusion-pipe 安装时直接拉 PyTorch 官方 wheel 索引
            <code className="text-foreground"> download.pytorch.org/whl/{"{cuda}"} </code>
            ；境内访问不畅时可在此填写镜像 base URL，会自动追加
            <code className="text-foreground"> /{"{cuda}"} </code> 后缀。留空使用官方源。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-[4px] border border-border/60 divide-y divide-border/40 overflow-hidden">
            {[
              {
                label: "官方（默认）",
                value: "",
              },
              {
                label: "清华 TUNA",
                value: "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/whl",
              },
              {
                label: "阿里云",
                value: "https://mirrors.aliyun.com/pytorch-wheels",
              },
            ].map((p) => {
              const isCurrent = draft.torch_index_url === p.value
              return (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => setDraft({ ...draft, torch_index_url: p.value })}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 text-xs text-left transition-colors",
                    isCurrent
                      ? "bg-primary/10 text-foreground"
                      : "hover:bg-muted/50 text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span className="flex-1 min-w-0">
                    <span className="block font-medium truncate">{p.label}</span>
                    {p.value && (
                      <span className="block text-[10px] font-mono text-muted-foreground/70 truncate">
                        {p.value}
                      </span>
                    )}
                  </span>
                  {isCurrent && (
                    <span className="rounded-[2px] bg-primary/10 text-primary px-1.5 py-0.5 text-[10px] uppercase tracking-[0.1em] shrink-0">
                      已选
                    </span>
                  )}
                </button>
              )
            })}
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">自定义</Label>
            <Input
              value={draft.torch_index_url}
              placeholder="https://mirror/.../pytorch-wheels（不含 /cuXXX 后缀）"
              onChange={(e) =>
                setDraft({ ...draft, torch_index_url: e.target.value })
              }
              className="font-mono"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Globe2 className="size-4 text-muted-foreground" />
            下载代理（模型下载与子进程）
          </CardTitle>
          <CardDescription>
            模型下载（HuggingFace / ModelScope）和训练 / 标注子进程使用的网络代理。支持
            <code className="text-foreground"> socks5h://user:pass@host:port </code>
            或
            <code className="text-foreground"> http://user:pass@host:port </code>
            格式。子进程会在未设置代理环境变量时注入
            <code className="text-foreground"> HTTPS_PROXY / HTTP_PROXY / ALL_PROXY </code>。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-start">
            <Label className="text-xs pt-2">代理地址</Label>
            <div className="space-y-1">
              <Input
                value={draft.download_proxy}
                placeholder="socks5h://user:pass@host:port"
                onChange={(e) =>
                  setDraft({ ...draft, download_proxy: e.target.value })
                }
                className="font-mono"
              />
              <p className="text-[11px] text-muted-foreground/80">
                SOCKS5 代理推荐使用 <code>socks5h://</code> 协议前缀（由代理端解析 DNS）。
                需要安装 <code>PySocks</code> 依赖才能使用 SOCKS 代理。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="size-4 text-muted-foreground" />
            Weights & Biases
          </CardTitle>
          <CardDescription>
            训练任务可上报指标到 wandb。在此填写 API Key,系统会注入
            <code className="text-foreground"> WANDB_API_KEY </code>
            环境变量,无需在 shell 里 export。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-start">
            <Label className="text-xs pt-2">API Key</Label>
            <div className="space-y-1">
              <Input
                type="password"
                value={draft.wandb_api_key}
                placeholder="wandb API Key（留空则不启用 wandb 上报）"
                onChange={(e) =>
                  setDraft({ ...draft, wandb_api_key: e.target.value })
                }
                className="font-mono"
              />
              <p className="text-[11px] text-muted-foreground/80">
                在 wandb.ai/authorize 获取。仅当配置的
                <code> monitoring.enable_wandb=true </code>
                时才会注入到训练子进程。
              </p>
            </div>
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-start">
            <Label className="text-xs pt-2">Base URL</Label>
            <div className="space-y-1">
              <Input
                value={draft.wandb_base_url}
                placeholder="https://wandb.your-domain.com（留空走 wandb.ai SaaS）"
                onChange={(e) =>
                  setDraft({ ...draft, wandb_base_url: e.target.value })
                }
                className="font-mono"
              />
              <p className="text-[11px] text-muted-foreground/80">
                自托管 W&amp;B Server 地址。同时作为
                <code> WANDB_BASE_URL </code>
                注入训练子进程,以及
                <code> wandb.Api(overrides=...) </code>
                的 base_url 用于「训练分析 → W&amp;B」拉取数据。
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
              pypi_index_url: draft.pypi_index_url || null,
              torch_index_url: draft.torch_index_url || null,
              download_proxy: draft.download_proxy || null,
              huggingface_token: draft.huggingface_token || null,
              wandb_api_key: draft.wandb_api_key || null,
              wandb_base_url: draft.wandb_base_url || null,
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
