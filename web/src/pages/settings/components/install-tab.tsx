import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  ArrowDownToLine,
  Download,
  Loader2,
  RefreshCw,
  XCircle,
} from "lucide-react"
import {
  api,
  useBootstrapStream,
  type BackendDescriptor,
  type BackendId,
  type BackendUpdateCheck,
  type BootstrapEvent,
  type TorchWheelOption,
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
import { AnimaModelDownloadCard, MsvcInstallCard } from "./install-action-cards"
import {
  EventLog,
  ProgressBar,
  STEP_PLANS,
  StatusBadge,
  StepList,
  computeStepStates,
  isRetryableStatus,
  isTerminalStatus,
  type StepDef,
} from "./install-progress"

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
    staleTime: 10_000,
  })

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 30_000,
  })

  const noProxy = !settingsQuery.data?.settings.github_proxy

  const torchOptionsQuery = useQuery({
    queryKey: ["backend-torch-options"],
    queryFn: api.getTorchOptions,
    staleTime: 60_000,
  })

  // Poll status as a fallback so the panel survives a page reload while an
  // install is mid-flight, and so we never miss a terminal frame the WS may
  // have sent before we attached.
  const statusQuery = useQuery({
    queryKey: ["backend-bootstrap-status"],
    queryFn: api.getBootstrapStatus,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
    staleTime: 750,
  })

  const rawStatus = statusQuery.data?.status ?? "idle"
  const isRunning = rawStatus === "running"
  const sessionBackend = statusQuery.data?.backend

  // The user's currently-selected backend — initialize it from the settings
  // default once both queries land. Use empty string until ready so the
  // base-ui Select stays controlled across the entire mount lifecycle.
  const [selected, setSelected] = useState<BackendId | "">("")
  const [selectedTorch, setSelectedTorch] = useState("")
  useEffect(() => {
    if (selected || !backendsQuery.data) return
    setSelected(backendsQuery.data.default)
  }, [backendsQuery.data, selected])

  const torchOptions = torchOptionsQuery.data?.options ?? []
  const recommendedTorch = useMemo(
    () => torchOptions.find((o) => o.recommended) ?? torchOptions.find((o) => o.compatible),
    [torchOptions],
  )
  useEffect(() => {
    if (selectedTorch || !recommendedTorch) return
    setSelectedTorch(torchOptionValue(recommendedTorch))
  }, [recommendedTorch, selectedTorch])
  const selectedTorchOption = useMemo(
    () => torchOptions.find((o) => torchOptionValue(o) === selectedTorch) ?? recommendedTorch,
    [recommendedTorch, selectedTorch, torchOptions],
  )

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
      return api.startBootstrap({
        backend,
        force: true,
        cuda: selectedTorchOption?.cuda,
        torch_version: selectedTorchOption?.torch_version,
        torchvision_version: selectedTorchOption?.torchvision_version,
        torch_override: backend === "anima_lora",
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
    },
  })

  const installDeps = useMutation({
    mutationFn: (backend: BackendId) => api.installDeps(backend),
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
    staleTime: 750,
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
    staleTime: 1_000,
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
    if (isTerminalStatus(status)) {
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

  // Backend git update detection — only for kohya and diffusion-pipe.
  const updateCheckEnabled =
    effective === "kohya" || effective === "diffusion-pipe"
  const updateCheckQuery = useQuery({
    queryKey: ["backend-update", effective],
    queryFn: () => api.checkBackendUpdate(effective as BackendId),
    enabled: updateCheckEnabled && !!descriptor?.ready,
    staleTime: 60_000,
    retry: false,
  })
  const applyUpdate = useMutation({
    mutationFn: () => api.updateBackend(effective as BackendId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-update", effective] })
      qc.invalidateQueries({ queryKey: ["backends"] })
    },
  })

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
                  当前未配置 GitHub 代理。可到
                  <strong className="text-foreground"> 网络加速 </strong>
                  标签页配置代理后再执行安装。
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
            <span className="text-xs text-muted-foreground">PyTorch</span>
            <Select
              value={selectedTorchOption ? torchOptionValue(selectedTorchOption) : ""}
              onValueChange={(value) => setSelectedTorch(value ?? "")}
              disabled={isRunning || torchOptions.length === 0}
            >
              <SelectTrigger className="w-72 text-xs font-mono h-8">
                <SelectValue placeholder="按驱动选择 PyTorch" />
              </SelectTrigger>
              <SelectContent align="start" alignItemWithTrigger={false} className="w-[360px]">
                {torchOptions.map((option) => (
                  <SelectItem
                    key={torchOptionValue(option)}
                    value={torchOptionValue(option)}
                    disabled={!option.compatible}
                  >
                    <span className="flex min-w-0 flex-col items-start gap-0.5">
                      <span className="truncate text-xs font-medium">
                        {option.label}
                        {option.recommended ? " · 默认" : ""}
                      </span>
                      <span className="truncate text-[10px] text-muted-foreground">
                        驱动最低 {option.min_driver} · {option.cuda}
                      </span>
                    </span>
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
                : isRetryableStatus(status)
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

          {selectedTorchOption && (
            <div
              className={cn(
                "rounded-[4px] border px-3 py-2 text-xs leading-relaxed",
                selectedTorchOption.compatible
                  ? "border-border/60 bg-muted/20 text-muted-foreground"
                  : "border-destructive/40 bg-destructive/5 text-destructive",
              )}
            >
              <span className="text-foreground">
                驱动 {torchOptionsQuery.data?.driver_version ?? "未检测到"} ·
                选择 {selectedTorchOption.torch_version} / {selectedTorchOption.cuda}
              </span>
              <span className="ml-2">{selectedTorchOption.reason}</span>
              {effective === "anima_lora" && (
                <span className="ml-2">
                  anima_lora 安装会在 <code>uv sync</code> 后设置 torch wheel。
                </span>
              )}
            </div>
          )}

          {updateCheckEnabled && descriptor?.ready && !isRunning && (
            <BackendUpdateCard
              data={updateCheckQuery.data}
              isFetching={updateCheckQuery.isFetching}
              isPending={applyUpdate.isPending}
              onCheck={() => updateCheckQuery.refetch()}
              onUpdate={() => applyUpdate.mutate()}
              error={applyUpdate.error as Error | null}
            />
          )}

          {effective === "anima_lora" && !isRunning && (
            <div className="rounded-[4px] border border-sky-500/40 bg-sky-500/5 px-3 py-2 text-xs text-sky-700 dark:text-sky-300 leading-relaxed">
              <strong className="text-foreground">anima_lora 已随 LoraHub 一起分发</strong>
              （<code className="text-foreground">external/anima_lora/</code>），
              点击「安装」会用
              <code className="text-foreground"> uv sync </code>
              在 <code className="text-foreground">external/anima_lora/.venv</code> 内
              创建一个<strong className="text-foreground">独立的</strong> CPython 3.13 venv,装好 torch 2.11/2.12 nightly +
              accelerate + diffusers 等依赖,与 LoraHub 主 venv 完全隔离。
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
              {isRetryableStatus(status) && lastError && (
                <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive break-all flex items-start gap-2">
                  <XCircle className="size-4 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-semibold not-italic">
                      {status === "interrupted" ? "安装中断" : "安装失败"}
                    </div>
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

function BackendUpdateCard({
  data,
  isFetching,
  isPending,
  onCheck,
  onUpdate,
  error,
}: {
  data: BackendUpdateCheck | undefined
  isFetching: boolean
  isPending: boolean
  onCheck: () => void
  onUpdate: () => void
  error: Error | null
}) {
  const hasUpdate = data?.update_available
  const hasError = data?.error

  return (
    <div
      className={cn(
        "rounded-[4px] border px-3 py-2.5 flex items-center gap-3",
        hasUpdate
          ? "border-sky-600/30 bg-sky-600/5"
          : hasError
            ? "border-amber-600/30 bg-amber-600/5"
            : "border-border/60 bg-muted/20",
      )}
    >
      <div className="flex-1 min-w-0 text-xs">
        {hasError && (
          <span className="text-amber-700 dark:text-amber-400">{data.error}</span>
        )}
        {hasUpdate && data && (
          <span className="text-sky-700 dark:text-sky-400">
            有 <span className="font-semibold">{data.commits_behind}</span> 个新提交可用
            <span className="ml-2 font-mono text-[10px] opacity-70">
              {data.current_sha.slice(0, 7)} → {data.remote_sha.slice(0, 7)}
            </span>
          </span>
        )}
        {!hasUpdate && !hasError && data && (
          <span className="text-muted-foreground">
            已是最新
            <span className="ml-2 font-mono text-[10px] opacity-70">
              {data.current_sha.slice(0, 7)} ({data.branch})
            </span>
          </span>
        )}
        {!data && !isFetching && (
          <span className="text-muted-foreground">点击检查仓库更新</span>
        )}
        {!data && isFetching && (
          <span className="text-muted-foreground">正在检查更新…</span>
        )}
      </div>
      <Button
        size="sm"
        variant="ghost"
        onClick={onCheck}
        disabled={isFetching}
        title="检查仓库更新"
        aria-label="检查仓库更新"
      >
        <RefreshCw className={cn("size-3", isFetching && "animate-spin")} />
        检查更新
      </Button>
      {hasUpdate && (
        <Button
          size="sm"
          variant="outline"
          onClick={onUpdate}
          disabled={isPending}
          className="shrink-0 gap-1.5 text-sky-700 dark:text-sky-400 border-sky-600/40 hover:bg-sky-600/10"
        >
          <ArrowDownToLine className="size-3" />
          {isPending ? "更新中…" : "更新"}
        </Button>
      )}
      {error && (
        <span className="text-[10px] text-destructive font-mono truncate max-w-[200px]">
          {error.message}
        </span>
      )}
    </div>
  )
}

function torchOptionValue(option: TorchWheelOption): string {
  return `${option.cuda}:${option.torch_version}:${option.torchvision_version}`
}
