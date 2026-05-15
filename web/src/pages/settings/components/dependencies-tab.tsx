/**
 * Dependencies tab — manages the portable Python runtime LoraHub uses as the
 * base for every backend's venv. The user can fetch a recommended CPython
 * build with one click; uv handles the actual download (it pulls
 * python-build-standalone from astral-sh's mirror, auto-selecting the right
 * artefact for the current OS / arch / libc).
 */
import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Cpu,
  Download,
  HardDrive,
  Loader2,
  Monitor,
  PackageCheck,
  Terminal,
  XCircle,
} from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

function PlatformBadge({
  system,
  machine,
}: {
  system: string
  machine: string
}) {
  const label = `${system} · ${machine}`
  return (
    <Badge variant="outline" className="rounded-[2px] gap-1.5 font-mono text-[10px]">
      <Monitor className="size-3" /> {label}
    </Badge>
  )
}

export function DependenciesTab() {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: ["runtime-python"],
    queryFn: api.getRuntimeStatus,
  })

  const [version, setVersion] = useState<string>("")
  // Sync the selector with the recommended default once the query lands.
  const effectiveVersion =
    version || status.data?.default_version || ""

  const install = useMutation({
    mutationFn: (v: string) => api.installRuntime(v),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runtime-python"] })
      qc.invalidateQueries({ queryKey: ["backends"] })
    },
  })

  const installing = install.isPending
  const error = install.error as Error | undefined

  const versions = useMemo(
    () => status.data?.recommended_versions ?? ["3.11", "3.12"],
    [status.data?.recommended_versions],
  )
  const installed = status.data?.installed ?? []
  const active = status.data?.active

  // uv lists every executable alias an interpreter exposes (python.exe,
  // python3.exe, python3.11.exe …) as separate entries. Collapse them to one
  // row per (version, arch, key) tuple, preferring the entry whose path
  // matches the active runtime so the "当前使用" badge stays anchored.
  const dedupedInstalled = useMemo(() => {
    const groups = new Map<string, typeof installed[number]>()
    for (const r of installed) {
      const key = `${r.implementation}-${r.version}-${r.arch}-${r.os}-${r.key}`
      const prev = groups.get(key)
      const matchesActive =
        active && active.version === r.version && active.path === r.path
      if (!prev) {
        groups.set(key, r)
        continue
      }
      const prevMatches =
        active && active.version === prev.version && active.path === prev.path
      // Anchor on the active path, otherwise prefer the shortest path
      // (typically the canonical executable instead of an alias symlink).
      if (matchesActive && !prevMatches) {
        groups.set(key, r)
      } else if (
        !prevMatches &&
        !matchesActive &&
        r.path &&
        r.path.length < prev.path.length
      ) {
        groups.set(key, r)
      }
    }
    return Array.from(groups.values())
  }, [installed, active])

  const hasMatch = installed.some((r) => r.version.startsWith(effectiveVersion))

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            便携 Python 运行时
          </CardTitle>
          <CardDescription>
            LoraHub 使用 uv 管理一份独立的 Python（python-build-standalone），
            作为 kohya / diffusion-pipe 等后端 venv 的基底，避免污染或依赖系统
            Python。下载按平台自动选择适配的构建。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {status.isLoading ? (
            <div className="text-sm text-muted-foreground">正在加载运行时状态…</div>
          ) : (
            <>
              <div className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-3 items-center">
                <span className="text-xs text-muted-foreground">当前平台</span>
                <div className="flex items-center gap-2 flex-wrap">
                  <PlatformBadge
                    system={status.data?.platform.system ?? ""}
                    machine={status.data?.platform.machine ?? ""}
                  />
                  <span className="text-[11px] text-muted-foreground/80 font-mono">
                    {status.data?.platform.release}
                  </span>
                </div>

                <span className="text-xs text-muted-foreground">安装目录</span>
                <div className="flex items-center gap-2 min-w-0">
                  <HardDrive className="size-3 text-muted-foreground shrink-0" />
                  <code
                    className="text-[11px] font-mono truncate"
                    title={status.data?.install_dir ?? ""}
                  >
                    {status.data?.install_dir}
                  </code>
                </div>

                <span className="text-xs text-muted-foreground">推荐版本</span>
                <div className="flex items-center gap-2">
                  <Select value={effectiveVersion} onValueChange={(v) => setVersion(v ?? "")}>
                    <SelectTrigger className="w-32 text-xs font-mono h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {versions.map((v) => (
                        <SelectItem key={v} value={v}>
                          Python {v}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="sm"
                    onClick={() => install.mutate(effectiveVersion)}
                    disabled={installing || !effectiveVersion}
                  >
                    {installing ? (
                      <Loader2 className="size-3 animate-spin" />
                    ) : (
                      <Download className="size-3" />
                    )}
                    {installing
                      ? "下载中…"
                      : hasMatch
                        ? "重新下载"
                        : "下载安装"}
                  </Button>
                </div>
              </div>

              {error && (
                <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive flex items-start gap-2 whitespace-pre-wrap break-all">
                  <XCircle className="size-4 shrink-0 mt-0.5" />
                  {error.message}
                </div>
              )}

              {install.isSuccess && !installing && (
                <div className="rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-400 flex items-center gap-2">
                  <PackageCheck className="size-4" />
                  已安装：
                  <code className="font-mono">{install.data?.installed.path}</code>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Terminal className="size-4 text-muted-foreground" />
            已安装的运行时
          </CardTitle>
          <CardDescription>
            uv 在本机已缓存的 Python 构建。后端安装时优先使用「
            {status.data?.default_version ?? "—"}
            」中匹配的版本作为 venv 基底。
          </CardDescription>
        </CardHeader>
        <CardContent>
          {dedupedInstalled.length === 0 ? (
            <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-6 text-center text-sm text-muted-foreground">
              暂无便携 Python。点击上方
              <span className="text-foreground font-medium">「下载安装」</span>
              开始首次配置。
            </div>
          ) : (
            <ul className="divide-y divide-border/40 rounded-[4px] border border-border/60 bg-muted/30">
              {dedupedInstalled.map((r) => {
                const isActive =
                  active && active.version === r.version && active.path === r.path
                return (
                  <li
                    key={`${r.version}-${r.path}`}
                    className={cn(
                      "px-3 py-2 flex items-center gap-3 text-xs font-mono",
                      isActive && "bg-primary/5",
                    )}
                  >
                    <Badge
                      variant={isActive ? "default" : "outline"}
                      className="rounded-[2px] uppercase text-[10px]"
                    >
                      {r.implementation || "cpython"}
                    </Badge>
                    <span className="font-semibold">{r.version}</span>
                    {r.arch && (
                      <span className="text-muted-foreground">{r.arch}</span>
                    )}
                    {r.os && (
                      <span className="text-muted-foreground">{r.os}</span>
                    )}
                    {isActive && (
                      <Badge
                        variant="secondary"
                        className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
                      >
                        当前使用
                      </Badge>
                    )}
                    <code
                      className="ml-auto text-muted-foreground truncate max-w-[28rem]"
                      title={r.path}
                    >
                      {r.path}
                    </code>
                  </li>
                )
              })}
            </ul>
          )}
        </CardContent>
      </Card>

      <div className="text-[11px] text-muted-foreground/80 px-1">
        · 便携 Python 不会修改系统环境变量；卸载只需删除安装目录即可。
        <br />· 若机器无法访问 astral-sh 默认镜像，可结合「网络加速」标签的 GitHub
        / HuggingFace 代理设置使用。
      </div>
    </div>
  )
}
