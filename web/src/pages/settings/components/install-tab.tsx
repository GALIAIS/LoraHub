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
  type BackendDescriptor,
  type BackendId,
  type BootstrapEvent,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
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
type StepDef = { id: string; label: string; match: RegExp }

const STEP_PLANS: Record<BackendId, StepDef[]> = {
  kohya: [
    { id: "clone", label: "克隆仓库", match: /^clone\s+kohya/i },
    { id: "venv", label: "创建虚拟环境", match: /^create venv/i },
    { id: "pip", label: "升级 pip / wheel", match: /^upgrade pip/i },
    { id: "torch", label: "安装 PyTorch", match: /^install torch/i },
    {
      id: "requirements",
      label: "安装 kohya requirements",
      match: /kohya requirements/i,
    },
    { id: "xformers", label: "安装 xformers", match: /^install xformers/i },
  ],
  "diffusion-pipe": [
    { id: "clone", label: "克隆仓库", match: /^clone\s+tdrussell/i },
    { id: "venv", label: "创建虚拟环境", match: /^create venv/i },
    { id: "pip", label: "升级 pip / wheel", match: /^upgrade pip/i },
    { id: "torch", label: "安装 PyTorch", match: /^install torch/i },
    {
      id: "requirements",
      label: "安装 diffusion-pipe requirements",
      match: /diffusion-pipe requirements/i,
    },
    { id: "deepspeed", label: "安装 DeepSpeed", match: /^install deepspeed/i },
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

  // Poll status as a fallback so the panel survives a page reload while an
  // install is mid-flight, and so we never miss a terminal frame the WS may
  // have sent before we attached.
  const statusQuery = useQuery({
    queryKey: ["backend-bootstrap-status"],
    queryFn: api.getBootstrapStatus,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
  })

  const status = statusQuery.data?.status ?? "idle"
  const isRunning = status === "running"
  const sessionBackend = statusQuery.data?.backend

  // The user's currently-selected backend — initialize it from the settings
  // default once both queries land. Use empty string until ready so the
  // base-ui Select stays controlled across the entire mount lifecycle.
  const [selected, setSelected] = useState<BackendId | "">("")
  useEffect(() => {
    if (selected || !backendsQuery.data) return
    setSelected(backendsQuery.data.default)
  }, [backendsQuery.data, selected])

  const [force, setForce] = useState(false)

  // While a session is running we always show the running backend, no matter
  // what the user previously picked — that's also what the action button
  // operates on, so the UX is unambiguous.
  const effective: BackendId | "" = isRunning
    ? (sessionBackend ?? selected)
    : selected

  // Stream live events while a session is active. Prefer the streamed
  // events when the WS has produced any (lower latency than the poll);
  // otherwise fall back to the buffered events from the polling query.
  const { events: streamedEvents } = useBootstrapStream(isRunning)
  const polled = statusQuery.data?.events ?? []
  const events: BootstrapEvent[] =
    streamedEvents.length > 0 ? streamedEvents : polled

  const start = useMutation({
    mutationFn: ({ backend, force }: { backend: BackendId; force: boolean }) => {
      console.info("[lorahub] startBootstrap", { backend, force })
      return api.startBootstrap({ backend, force })
    },
    onSuccess: () => {
      // Keep `force` flipped on so the user can re-trigger directly if the
      // install hits an error mid-way; they'll explicitly turn it off when
      // they no longer want to overwrite.
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
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

  // Friendlier 409 conflict message: server reports "target ... is not empty".
  const startConflict =
    startError && /409|not empty/i.test(startError.message)
      ? startError.message
      : null

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">安装训练后端</CardTitle>
          <CardDescription>
            克隆仓库、创建 venv，并安装 PyTorch 与依赖。与命令行 {" "}
            <code className="text-foreground">lorahub bootstrap-*</code> 等价；
            安装在后台运行，请保持本页打开。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-muted-foreground">后端</span>
            <Select
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
            <div className="flex items-center gap-2">
              <Switch
                id="install-force"
                checked={force}
                onCheckedChange={setForce}
                disabled={isRunning}
              />
              <Label htmlFor="install-force" className="text-xs cursor-pointer">
                强制重装（清空目标目录）
              </Label>
            </div>
            <Button
              size="sm"
              disabled={
                isRunning ||
                start.isPending ||
                !effective ||
                !backendsQuery.data
              }
              onClick={() =>
                effective && start.mutate({ backend: effective as BackendId, force })
              }
            >
              {isRunning ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {isRunning
                ? "安装中…"
                : force
                  ? "强制重装"
                  : status === "failed"
                    ? "重试安装"
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

          {startConflict && (
            <div className="rounded-[4px] border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
              <AlertTriangle className="size-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">目标目录已存在</div>
                <div className="font-mono break-all mt-0.5">{startConflict}</div>
                <div className="mt-1">
                  打开 <span className="font-semibold">「强制重装」</span> 后再次点击
                  「安装」可清空原目录。
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
