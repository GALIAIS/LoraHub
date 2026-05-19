import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  Check,
  CircleDashed,
  Download,
  Loader2,
  XCircle,
} from "lucide-react"
import {
  api,
  useBootstrapStream,
  type AnimaLoraBackendStatus,
  type AnimaModelDownloadStatus,
  type BackendDescriptor,
  type BackendId,
  type BootstrapEvent,
  type MsvcInstallStatus,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { BackendStatusCard } from "./backend-status-card"

// Per-backend ordered step plan + a regex matching each step's progress
// message. Keep these in sync with the corresponding installer.bootstrap()
// implementation in lorahub/core/backends/<id>/installer.py.
//
// The patterns are anchored with a leading ^ wherever feasible so that
// indented git-progress lines emitted by run_step (`  Receiving objects:
// 23%`) never accidentally match. Matching is case-insensitive.
type StepDef = { id: string; label: string; match: RegExp }

const STEP_PLANS: Record<BackendId, StepDef[]> = {
  kohya: [
    { id: "clone", label: "克隆仓库", match: /^clone\s+kohya-ss\//i },
    // installer prints `uv venv -> .../venv (python=...)` (see
    // lorahub/core/toolchain/uv.py::create_venv). We also accept a
    // permissive form so a future label change still highlights this step.
    { id: "venv", label: "创建虚拟环境", match: /^(uv\s+venv\b|create\s+venv)/i },
    { id: "pip", label: "升级 pip / wheel", match: /^upgrade\s+pip/i },
    { id: "torch", label: "安装 PyTorch", match: /^install\s+torch/i },
    {
      id: "requirements",
      label: "安装 kohya requirements",
      match: /kohya\s+requirements/i,
    },
    { id: "xformers", label: "安装 xformers", match: /^install\s+xformers/i },
  ],
  "diffusion-pipe": [
    { id: "clone", label: "克隆仓库", match: /^clone\s+tdrussell\//i },
    { id: "venv", label: "创建虚拟环境", match: /^(uv\s+venv\b|create\s+venv)/i },
    { id: "pip", label: "升级 pip / wheel", match: /^upgrade\s+pip/i },
    { id: "torch", label: "安装 PyTorch", match: /^install\s+torch/i },
    {
      id: "requirements",
      label: "安装 diffusion-pipe requirements",
      match: /diffusion-pipe\s+requirements/i,
    },
    { id: "deepspeed", label: "安装 DeepSpeed", match: /^install\s+deepspeed/i },
  ],
  // anima_lora ships vendored under external/anima_lora — no clone
  // needed. The single install step is `uv sync` against the bundled
  // pyproject.toml + uv.lock, which materialises a dedicated `.venv`
  // (CPython 3.13 + torch nightly + accelerate + diffusers). Surface
  // it as one step so the progress UI lines up with kohya / dp.
  anima_lora: [
    { id: "sync", label: "uv sync (创建 .venv + 装依赖)", match: /^uv\s+sync/i },
  ],
}

type StepState = "pending" | "running" | "succeeded" | "failed"

function computeStepStates(
  plan: StepDef[],
  events: BootstrapEvent[],
  status: string,
): { states: StepState[]; current: number } {
  const states: StepState[] = plan.map(() => "pending")
  let current = -1
  for (const ev of events) {
    if (ev.level !== "info") continue
    const idx = plan.findIndex((s) => s.match.test(ev.message))
    if (idx < 0) continue
    if (current >= 0 && current < plan.length) states[current] = "succeeded"
    current = idx
    states[idx] = "running"
  }
  if (current >= 0) {
    if (status === "succeeded") {
      // Mark every step done; some plans skip optional ones (xformers off etc.)
      for (let i = 0; i < states.length; i += 1) {
        if (states[i] === "pending" || states[i] === "running") {
          states[i] = "succeeded"
        }
      }
    } else if (status === "failed") {
      states[current] = "failed"
    }
  } else if (status === "succeeded") {
    // No info events arrived (shouldn't happen in practice); still mark done.
    for (let i = 0; i < states.length; i += 1) states[i] = "succeeded"
  }
  return { states, current }
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "succeeded")
    return <Check className="size-3.5 text-emerald-600 dark:text-emerald-400" />
  if (state === "running")
    return <Loader2 className="size-3.5 animate-spin text-amber-600 dark:text-amber-400" />
  if (state === "failed") return <XCircle className="size-3.5 text-destructive" />
  return <CircleDashed className="size-3.5 text-muted-foreground/60" />
}

function StepList({
  plan,
  states,
}: {
  plan: StepDef[]
  states: StepState[]
}) {
  return (
    <ol className="rounded-[4px] border border-border/60 bg-muted/30 divide-y divide-border/40">
      {plan.map((s, i) => {
        const state = states[i]
        return (
          <li
            key={s.id}
            className={cn(
              "px-3 py-2 flex items-center gap-3 text-xs",
              state === "running" && "bg-amber-500/5",
              state === "failed" && "bg-destructive/5",
              state === "succeeded" && "text-muted-foreground",
            )}
          >
            <StepIcon state={state} />
            <span className="flex-1">{s.label}</span>
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
              {state === "pending" ? "等待" : state === "running" ? "进行中" : state === "succeeded" ? "完成" : "失败"}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

function ProgressBar({
  done,
  total,
  failed,
}: {
  done: number
  total: number
  failed: boolean
}) {
  const pct = total > 0 ? Math.min(100, Math.max(0, (done / total) * 100)) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>
          已完成 {done} / {total}
        </span>
        <span className="font-mono tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-[1px] bg-muted/40 overflow-hidden">
        <div
          className={cn(
            "h-full transition-[width] duration-300",
            failed ? "bg-destructive" : "bg-emerald-500",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "succeeded"
      ? "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/5"
      : status === "failed"
        ? "text-destructive border-destructive/40 bg-destructive/5"
        : status === "running"
          ? "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/5"
          : "text-muted-foreground border-border/60 bg-muted/30"
  const label =
    status === "succeeded"
      ? "已完成"
      : status === "failed"
        ? "失败"
        : status === "running"
          ? "运行中"
          : "空闲"
  return (
    <span
      className={cn(
        "px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] rounded-[3px] border font-mono",
        tone,
      )}
    >
      {label}
    </span>
  )
}

function EventLog({ events }: { events: BootstrapEvent[] }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-zinc-950 dark:bg-zinc-900 max-h-72 overflow-y-auto">
      <ol className="divide-y divide-zinc-800/60 font-mono text-[11px]">
        {events.map((ev, idx) => (
          <li
            key={`${ev.ts}-${idx}`}
            className={cn(
              "px-3 py-1.5 flex items-start gap-2",
              ev.level === "error"
                ? "text-red-400 bg-red-950/40 border-l-2 border-l-red-500"
                : ev.level === "done"
                  ? "text-emerald-400"
                  : "text-zinc-100",
            )}
          >
            <span className="text-[9px] uppercase tracking-[0.18em] text-zinc-500 w-12 shrink-0 pt-0.5">
              {ev.level}
            </span>
            <span className="break-all">{ev.message}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

/**
 * One-click download for the three multi-GB anima checkpoints (DiT,
 * Qwen3 TE, Qwen Image VAE). Renders only when ``uv sync`` finished
 * but the files aren't on disk yet — i.e. the venv is ready and the
 * user only needs the weights to start training. Shows live progress
 * while a download is running and a clear "all set" state once done.
 */
function AnimaModelDownloadCard({
  missing,
  status,
  isPending,
  onDownload,
  error,
}: {
  missing: string[]
  status: AnimaModelDownloadStatus | undefined
  isPending: boolean
  onDownload: () => void
  error: string | null
}) {
  const isRunning = status?.status === "running"
  const failed = status?.status === "failed"
  const succeeded = status?.status === "succeeded" && missing.length === 0
  const percent = status?.percent ?? 0
  const filesDone = status?.files_done ?? 0
  const filesTotal = status?.files_total ?? missing.length
  const lastEvent = status?.events?.[status.events.length - 1]

  return (
    <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-3 space-y-2.5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="flex-1 text-xs text-amber-700 dark:text-amber-400">
          <div className="font-semibold text-foreground">
            anima 模型未就绪
          </div>
          <div className="mt-0.5 leading-relaxed">
            训练 / 推理需要 3 个 safetensors 检查点（DiT 基模型、Qwen3 文本编码器、
            Qwen Image VAE），从 HuggingFace
            <code className="mx-1 text-foreground">circlestone-labs/Anima</code>
            下载约 <strong className="text-foreground">14 GB</strong>，存放到项目根
            <code className="mx-1 text-foreground">models/</code>
            目录。
          </div>
          {missing.length > 0 && !succeeded && (
            <ul className="mt-1.5 ml-2 font-mono text-[11px] space-y-0.5 text-muted-foreground">
              {missing.map((f) => (
                <li key={f}>· {f}</li>
              ))}
            </ul>
          )}
        </div>
        <Button
          size="sm"
          variant={succeeded ? "outline" : "default"}
          disabled={isRunning || isPending || succeeded}
          onClick={onDownload}
        >
          {isRunning || isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : succeeded ? (
            <Check className="size-3" />
          ) : (
            <Download className="size-3" />
          )}
          {isRunning
            ? "下载中…"
            : succeeded
              ? "已完成"
              : failed
                ? "重试"
                : "下载模型"}
        </Button>
      </div>

      {(isRunning || filesDone > 0) && filesTotal > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>
              已完成 {filesDone} / {filesTotal} 文件
            </span>
            <span className="font-mono tabular-nums">{percent.toFixed(0)}%</span>
          </div>
          <div className="h-1.5 rounded-[1px] bg-muted/40 overflow-hidden">
            <div
              className={cn(
                "h-full transition-[width] duration-300",
                failed ? "bg-destructive" : "bg-emerald-500",
              )}
              style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
            />
          </div>
          {lastEvent && (
            <div className="font-mono text-[10px] text-muted-foreground/80 break-all">
              {lastEvent.message}
            </div>
          )}
        </div>
      )}

      {(failed || error) && (
        <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-2.5 py-1.5 text-[11px] font-mono text-destructive break-all">
          {status?.error || error}
        </div>
      )}
    </div>
  )
}

/**
 * Visual Studio Build Tools (MSVC) install card.
 *
 * Renders only when the anima_lora venv is ready but Windows MSVC
 * is missing. anima's ``--torch_compile`` path drives PyTorch
 * Inductor to JIT through triton-windows; without ``cl.exe`` the
 * trainer crashes inside the first compile pass with a TypeError
 * from triton's MSVC discovery — a failure mode that has no
 * obvious connection to the user's recipe and is hard to diagnose
 * after the fact. The button shells out to ``winget install
 * Microsoft.VisualStudio.2022.BuildTools`` with the C++ workload +
 * Win 11 SDK pre-selected; the tail of winget's output is mirrored
 * into the log block below the button so the user can see when
 * MSI installers are still spinning vs actually wedged.
 */
function MsvcInstallCard({
  detection,
  status,
  isPending,
  onInstall,
  error,
}: {
  detection: AnimaLoraBackendStatus["msvc"]
  status: MsvcInstallStatus | undefined
  isPending: boolean
  onInstall: () => void
  error: string | null
}) {
  const isRunning = status?.status === "running"
  const failed = status?.status === "failed"
  const succeeded = status?.msvc?.ok || status?.status === "succeeded"
  const log = status?.log ?? []
  const lastLine = log.length > 0 ? log[log.length - 1] : null

  return (
    <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-3 space-y-2.5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="flex-1 text-xs text-amber-700 dark:text-amber-400">
          <div className="font-semibold text-foreground">
            缺少 Visual Studio Build Tools
          </div>
          <div className="mt-0.5 leading-relaxed">
            anima_lora 训练默认开启
            <code className="mx-1 text-foreground">torch.compile</code>
            ，PyTorch Inductor 需要通过 triton-windows 调用
            <code className="mx-1 text-foreground">cl.exe</code>
            。否则首次编译就会崩溃，无法继续训练。
          </div>
          <div className="mt-1 leading-relaxed">
            点击下方按钮调用
            <code className="mx-1 text-foreground">winget</code>
            自动安装
            <strong className="mx-0.5 text-foreground">
              Build Tools for Visual Studio 2022
            </strong>
            （含 C++ 工作负载与 Windows 11 SDK，约 1.5–2 GB）。不会安装完整的 Visual Studio IDE。
          </div>
          {detection.reason && (
            <div className="mt-1 font-mono text-[10px] text-muted-foreground/80 break-all">
              {detection.reason}
            </div>
          )}
          {!detection.winget_available && (
            <div className="mt-1.5 text-[11px]">
              <strong className="text-foreground">winget 不可用</strong>
              ，需要手动下载安装：
              <a
                href="https://aka.ms/vs/17/release/vs_BuildTools.exe"
                target="_blank"
                rel="noreferrer"
                className="ml-1 underline"
              >
                vs_BuildTools.exe
              </a>
            </div>
          )}
        </div>
        <Button
          size="sm"
          variant={succeeded ? "outline" : "default"}
          disabled={
            isRunning || isPending || succeeded || !detection.winget_available
          }
          onClick={onInstall}
        >
          {isRunning || isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : succeeded ? (
            <Check className="size-3" />
          ) : (
            <Download className="size-3" />
          )}
          {isRunning
            ? "安装中…"
            : succeeded
              ? "已安装"
              : failed
                ? "重试"
                : "一键安装"}
        </Button>
      </div>

      {(isRunning || lastLine) && (
        <div className="space-y-1">
          {lastLine && (
            <div className="font-mono text-[10px] text-muted-foreground/80 break-all">
              {lastLine}
            </div>
          )}
          {log.length > 1 && (
            <details className="text-[10px] text-muted-foreground/70">
              <summary className="cursor-pointer select-none">
                查看完整日志（{log.length} 行）
              </summary>
              <pre className="mt-1 max-h-48 overflow-auto rounded-[3px] border border-border/60 bg-muted/20 px-2 py-1.5 font-mono text-[10px] text-foreground/70">
                {log.join("\n")}
              </pre>
            </details>
          )}
        </div>
      )}

      {(failed || error) && (
        <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-2.5 py-1.5 text-[11px] font-mono text-destructive break-all">
          {status?.error || error}
        </div>
      )}
    </div>
  )
}

/**
 * One-click install panel. The user picks a backend, hits 安装, and the
 * server kicks off the registry-driven bootstrap runner. Because the server
 * keeps a single bootstrap session at a time, while one is in flight we
 * lock the selector to whatever backend is actually running.
 */
export function InstallTab() {
  const qc = useQueryClient()

  const backendsQuery = useQuery({
    queryKey: ["backends"],
    queryFn: api.listBackends,
  })

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const noProxy = !settingsQuery.data?.settings.github_proxy

  // Poll status as a fallback so the panel survives a page reload while an
  // install is mid-flight, and so we never miss a terminal frame the WS may
  // have sent before we attached.
  const statusQuery = useQuery({
    queryKey: ["backend-bootstrap-status"],
    queryFn: api.getBootstrapStatus,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
  })

  const rawStatus = statusQuery.data?.status ?? "idle"
  const isRunning = rawStatus === "running"
  const sessionBackend = statusQuery.data?.backend

  // The user's currently-selected backend — initialize it from the settings
  // default once both queries land. Use empty string until ready so the
  // base-ui Select stays controlled across the entire mount lifecycle.
  const [selected, setSelected] = useState<BackendId | "">("")
  useEffect(() => {
    if (selected || !backendsQuery.data) return
    setSelected(backendsQuery.data.default)
  }, [backendsQuery.data, selected])

  // While a session is running we always show the running backend, no matter
  // what the user previously picked — that's also what the action button
  // operates on, so the UX is unambiguous.
  const effective: BackendId | "" = isRunning
    ? (sessionBackend ?? selected)
    : selected

  const backendOptions = useMemo(
    () =>
      (backendsQuery.data?.backends ?? []).map((b) => ({
        value: b.id,
        label: b.name,
      })),
    [backendsQuery.data?.backends],
  )

  // Stream live events while a session is active. Prefer the streamed
  // events when the WS has produced any (lower latency than the poll);
  // otherwise fall back to the buffered events from the polling query.
  const { events: streamedEvents } = useBootstrapStream(isRunning)
  const polled = statusQuery.data?.events ?? []
  const rawEvents: BootstrapEvent[] =
    streamedEvents.length > 0 ? streamedEvents : polled

  // Hide stale events from a different backend's session. The polled
  // status endpoint always returns the most recent session regardless
  // of which backend the user has currently selected; without this
  // gate, switching from "diffusion-pipe (failed)" to "anima_lora"
  // would still show the dp error in the event log + status badge.
  // We only render events when (a) a session is actively running for
  // the selected backend, OR (b) the most recent finished session
  // matches the selected backend.
  const sameSession = sessionBackend && sessionBackend === selected
  const events: BootstrapEvent[] = sameSession ? rawEvents : []
  // Same gating for the status badge so a "failed" tag from another
  // backend doesn't bleed onto the freshly-selected one.
  const status = sameSession ? rawStatus : "idle"

  const start = useMutation({
    mutationFn: (backend: BackendId) => {
      // Always force-overwrite. Users have repeatedly tripped on the
      // "target not empty" 409 because the install panel's whole point is
      // to wipe and re-install — there's no other meaningful intent.
      console.info("[lorahub] startBootstrap", { backend, force: true })
      return api.startBootstrap({ backend, force: true })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
    },
  })

  const installDeps = useMutation({
    mutationFn: (backend: BackendId) => {
      console.info("[lorahub] installDeps", { backend })
      return api.installDeps(backend)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["backends"] })
    },
  })

  // Anima base / TE / VAE checkpoints — separate download flow because
  // the files are multi-GB and shouldn't ride the bootstrap pipeline.
  const animaModelStatus = useQuery({
    queryKey: ["anima-model-download-status"],
    queryFn: api.getAnimaModelDownloadStatus,
    enabled: effective === "anima_lora",
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
  })

  const downloadAnimaModels = useMutation({
    mutationFn: () => api.startAnimaModelDownload(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["anima-model-download-status"] })
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["backends"] })
    },
  })

  // MSVC build tools — Windows-only. The poll has a tighter cadence
  // than the model download because winget shells out to MSI installers
  // that can take 5–15 min, and a status flip from "running" to
  // "succeeded" / "failed" is the cue to refresh the backend catalog
  // so ``msvc.ok`` lights up.
  const msvcInstallStatus = useQuery({
    queryKey: ["msvc-install-status"],
    queryFn: api.getMsvcInstallStatus,
    enabled: effective === "anima_lora",
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 2000 : false,
  })

  const installMsvc = useMutation({
    mutationFn: () => api.startMsvcInstall(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["msvc-install-status"] })
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["backends"] })
    },
  })

  // When an install transitions to a terminal state, refresh the backend
  // catalog so the user sees their new checkout immediately.
  useEffect(() => {
    if (status === "succeeded" || status === "failed") {
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["backends"] })
      qc.invalidateQueries({ queryKey: ["health"] })
    }
  }, [status, qc])

  const startError = start.error as Error | null
  const lastError = events.find((e) => e.level === "error")

  const descriptor: BackendDescriptor | undefined = useMemo(() => {
    if (!backendsQuery.data || !effective) return undefined
    return backendsQuery.data.backends.find((b) => b.id === effective)
  }, [backendsQuery.data, effective])

  const isOtherSessionRunning =
    isRunning && sessionBackend !== undefined && sessionBackend !== selected

  // Step plan + state derivation for the chosen backend.
  const plan: StepDef[] = effective ? STEP_PLANS[effective as BackendId] ?? [] : []
  const { states } = computeStepStates(plan, events, status)
  const doneCount = states.filter((s) => s === "succeeded").length
  const failedCount = states.filter((s) => s === "failed").length
  const showSteps = plan.length > 0 && (events.length > 0 || isRunning)

  // The only 409 case left after we always send force=true is "failed to
  // clear" — surface that clearly so the user knows to free the locks.
  const startConflict =
    startError && /failed to clear|some files may be locked/i.test(startError.message)
      ? startError.message
      : null

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">安装训练后端</CardTitle>
          <CardDescription>
            克隆仓库、创建 venv，并安装 PyTorch 与依赖。
            <strong className="text-foreground">已存在的目标目录会被清空后重装</strong>。
            与命令行 <code className="text-foreground">lorahub bootstrap-*</code> 等价；
            安装在后台运行，请保持本页打开。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {noProxy && (
            <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">GitHub 代理未配置</div>
                <div className="mt-0.5">
                  国内直连 GitHub 克隆仓库可能极慢或超时。建议先到
                  <strong className="text-foreground"> 网络加速 </strong>
                  标签页配置 GitHub 代理（推荐
                  <code className="text-foreground"> https://gh-proxy.org </code>
                  ），再执行安装。
                </div>
              </div>
            </div>
          )}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-muted-foreground">后端</span>
            <Select
              items={backendOptions}
              value={effective || ""}
              onValueChange={(v) => setSelected(v as BackendId)}
              disabled={isRunning || !backendsQuery.data}
            >
              <SelectTrigger className="w-64 text-xs font-mono h-8">
                <SelectValue placeholder="选择要安装的后端" />
              </SelectTrigger>
              <SelectContent>
                {(backendsQuery.data?.backends ?? []).map((b) => (
                  <SelectItem key={b.id} value={b.id}>
                    {b.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              disabled={
                isRunning ||
                start.isPending ||
                !effective ||
                !backendsQuery.data
              }
              onClick={() => effective && start.mutate(effective as BackendId)}
            >
              {isRunning ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {isRunning
                ? "安装中…"
                : status === "failed"
                  ? "重试安装"
                  : descriptor?.ready
                    ? "重新安装"
                    : "安装"}
            </Button>
            <StatusBadge status={status} />
            {isOtherSessionRunning && (
              <span className="text-xs text-amber-600 dark:text-amber-400">
                另一个后端（
                <code className="font-mono">{sessionBackend}</code>
                ）正在安装，请等待完成。
              </span>
            )}
          </div>

          {effective === "anima_lora" && !isRunning && (
            <div className="rounded-[4px] border border-sky-500/40 bg-sky-500/5 px-3 py-2 text-xs text-sky-700 dark:text-sky-300 leading-relaxed">
              <strong className="text-foreground">anima_lora 已随 LoraHub 一起分发</strong>
              （<code className="text-foreground">external/anima_lora/</code>），
              无需克隆。点击「安装」会用
              <code className="text-foreground"> uv sync </code>
              在 <code className="text-foreground">external/anima_lora/.venv</code> 内
              创建一个<strong className="text-foreground">独立的</strong> CPython 3.13 venv,装好 torch 2.11/2.12 nightly +
              accelerate + diffusers 等依赖,与 LoraHub 主 venv 完全隔离。
              首次安装大约下载 6-8 GB（torch + CUDA wheels）。
            </div>
          )}

          {effective === "anima_lora" &&
            descriptor?.ready &&
            descriptor.status.id === "anima_lora" &&
            !descriptor.status.models_ok && (
              <AnimaModelDownloadCard
                missing={descriptor.status.missing_models}
                status={animaModelStatus.data}
                isPending={downloadAnimaModels.isPending}
                onDownload={() => downloadAnimaModels.mutate()}
                error={
                  downloadAnimaModels.error instanceof Error
                    ? downloadAnimaModels.error.message
                    : null
                }
              />
            )}

          {effective === "anima_lora" &&
            descriptor?.ready &&
            descriptor.status.id === "anima_lora" &&
            descriptor.status.msvc.platform_relevant &&
            !descriptor.status.msvc.ok && (
              <MsvcInstallCard
                detection={descriptor.status.msvc}
                status={msvcInstallStatus.data}
                isPending={installMsvc.isPending}
                onInstall={() => installMsvc.mutate()}
                error={
                  installMsvc.error instanceof Error
                    ? installMsvc.error.message
                    : null
                }
              />
            )}

          {descriptor && !descriptor.status.requirements_ok && descriptor.status.python_ok && !isRunning && (
            <div className="flex items-center gap-3 rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-2">
              <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0" />
              <span className="text-xs text-amber-700 dark:text-amber-400 flex-1">
                检测到 {descriptor.status.missing_requirements.length} 个依赖未安装
              </span>
              <Button
                size="sm"
                variant="outline"
                disabled={isRunning || installDeps.isPending}
                onClick={() => effective && installDeps.mutate(effective as BackendId)}
              >
                {installDeps.isPending ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Download className="size-3" />
                )}
                安装依赖
              </Button>
            </div>
          )}

          {startConflict && (
            <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">无法清空目标目录</div>
                <div className="font-mono break-all mt-0.5">{startConflict}</div>
                <div className="mt-1">
                  常见原因：进程仍持有该目录中的文件（如编辑器、文件管理器、
                  早先的 venv）。关闭它们后再点「重试安装」，或手动删除目录。
                </div>
              </div>
            </div>
          )}

          {startError && !startConflict && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive break-all">
              {startError.message}
            </div>
          )}

          {showSteps && (
            <div className="space-y-3">
              <ProgressBar
                done={doneCount}
                total={plan.length}
                failed={failedCount > 0}
              />
              <StepList plan={plan} states={states} />
              {status === "failed" && lastError && (
                <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive break-all flex items-start gap-2">
                  <XCircle className="size-4 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-semibold not-italic">安装失败</div>
                    <div className="mt-0.5 whitespace-pre-wrap">
                      {lastError.message}
                    </div>
                    {plan.length - doneCount > 0 && (
                      <div className="mt-1 text-muted-foreground">
                        还剩 {plan.length - doneCount} 个步骤未完成。可勾选
                        「强制重装」并重试。
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {descriptor && (
            <BackendStatusCard
              descriptor={descriptor}
              isDefault={descriptor.id === backendsQuery.data?.default}
              compact
            />
          )}

          {events.length > 0 && <EventLog events={events} />}
        </CardContent>
      </Card>
    </div>
  )
}
