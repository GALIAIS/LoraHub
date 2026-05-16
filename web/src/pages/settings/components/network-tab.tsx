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
  modelscope_enabled: boolean
  modelscope_token: string
  pypi_index_url: string
  download_proxy: string
}

function buildDraft(s: SettingsState): Draft {
  return {
    github_proxy: s.github_proxy ?? "",
    huggingface_endpoint: s.huggingface_endpoint ?? "",
    modelscope_enabled: s.modelscope_enabled,
    modelscope_token: s.modelscope_token ?? "",
    pypi_index_url: s.pypi_index_url ?? "",
    download_proxy: s.download_proxy ?? "",
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
    draft.modelscope_enabled !== saved.modelscope_enabled ||
    draft.modelscope_token !== (saved.modelscope_token ?? "") ||
    draft.pypi_index_url !== (saved.pypi_index_url ?? "") ||
    draft.download_proxy !== (saved.download_proxy ?? "")

  const githubPresets = presetsQuery.data?.github_proxy ?? []
  const hfPresets = presetsQuery.data?.huggingface ?? []
  const pypiPresets = presetsQuery.data?.pypi ?? []

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

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
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

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Globe2 className="size-4 text-muted-foreground" />
            下载代理（模型下载）
          </CardTitle>
          <CardDescription>
            模型下载（HuggingFace / ModelScope）时使用的网络代理。支持
            <code className="text-foreground"> socks5h://user:pass@host:port </code>
            或
            <code className="text-foreground"> http://user:pass@host:port </code>
            格式。留空表示直连。
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
              download_proxy: draft.download_proxy || null,
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
