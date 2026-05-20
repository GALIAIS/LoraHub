/**
 * Terminal page — venv-scoped command runner.
 *
 * The page composes three subcomponents:
 *
 *   - <BackendPicker>: choose which backend the next command runs in.
 *     Lists every known backend and disables the unavailable ones.
 *   - <PseudoTerminal>: the scrollback view + input row. Owns command
 *     history (localStorage), aborts the in-flight stream when the user
 *     starts a new command, renders coloured stdout/stderr.
 *   - <QuickCommands>: a row of one-click presets ("pip list", "pip
 *     freeze", ...) that splice into the input.
 *
 * State lives at the page level so switching backends clears scrollback,
 * and the abort controller for any in-flight command is owned by the
 * page (not the input row) so a backend switch can interrupt a running
 * pip install cleanly.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { ShieldAlert, Terminal as TerminalIcon } from "lucide-react"
import { toast } from "sonner"
import {
  terminalExec,
  terminalListSessions,
  type TerminalEnvironment,
  type TerminalEvent,
} from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { BackendPicker } from "./components/backend-picker"
import { PseudoTerminal } from "./components/pseudo-terminal"
import { QuickCommands } from "./components/quick-commands"
import { type TerminalLine } from "./components/types"

const HISTORY_STORAGE_KEY = "lorahub.terminal.command-history"
const MAX_HISTORY = 100

export function TerminalPage() {
  const sessionsQuery = useQuery({
    queryKey: ["terminal", "sessions"],
    queryFn: terminalListSessions,
    staleTime: 30_000,
  })

  const sessions = sessionsQuery.data?.backends ?? []
  const defaultBackend = sessionsQuery.data?.default_backend ?? "kohya"
  const unrestricted = sessionsQuery.data?.unrestricted ?? false
  const [backendId, setBackendId] = useState<string | null>(null)

  // Default to settings.default_backend the first time we get a list.
  // We don't auto-update on every refetch so the user's manual pick
  // sticks — re-pinning would be confusing if they're mid-investigation
  // when settings changes elsewhere.
  useEffect(() => {
    if (backendId == null && sessions.length > 0) {
      const initial =
        sessions.find((s) => s.backend_id === defaultBackend && s.ready) ??
        sessions.find((s) => s.ready) ??
        sessions[0]
      if (initial) setBackendId(initial.backend_id)
    }
  }, [sessions, defaultBackend, backendId])

  const activeSession: TerminalEnvironment | null = useMemo(
    () => sessions.find((s) => s.backend_id === backendId) ?? null,
    [sessions, backendId],
  )

  const [lines, setLines] = useState<TerminalLine[]>([])
  const [running, setRunning] = useState(false)
  const [input, setInput] = useState("")
  const abortRef = useRef<AbortController | null>(null)

  // Persist + recall command history across reloads. Trimmed to the
  // last MAX_HISTORY commands so localStorage doesn't bloat.
  const [history, setHistory] = useState<string[]>(() => {
    if (typeof window === "undefined") return []
    try {
      const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY)
      if (!raw) return []
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed.filter((s) => typeof s === "string") : []
    } catch {
      return []
    }
  })
  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify(history.slice(-MAX_HISTORY)),
    )
  }, [history])

  // Switching backends clears scrollback and aborts any running stream.
  // Without this the user could mistakenly attribute a pip-list output
  // from kohya to anima_lora after switching.
  useEffect(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setRunning(false)
    setLines([])
  }, [backendId])

  const appendLine = useCallback((line: TerminalLine) => {
    setLines((prev) => [...prev, line])
  }, [])

  const runCommand = useCallback(
    async (command: string) => {
      const trimmed = command.trim()
      if (!trimmed || running || !activeSession) return
      if (!activeSession.ready) {
        toast.error("此后端尚未安装", {
          description: "请到 设置 → 后端管理 完成安装后再使用终端。",
        })
        return
      }
      // Echo the command into the scrollback first so the user has a
      // visual anchor while output streams in.
      appendLine({
        kind: "prompt",
        prompt: activeSession.prompt,
        text: trimmed,
      })
      setHistory((prev) =>
        prev.length > 0 && prev[prev.length - 1] === trimmed
          ? prev
          : [...prev, trimmed].slice(-MAX_HISTORY),
      )
      setInput("")

      const controller = new AbortController()
      abortRef.current = controller
      setRunning(true)
      // Track output so we can detect "exit non-zero with no output" —
      // a common failure shape on Windows when the venv python is
      // missing a runtime dep and produces no stdout/stderr at all.
      let sawOutput = false
      let lastArgv: string[] | null = null
      let lastCwd: string | null = null
      try {
        await terminalExec(
          { backend_id: activeSession.backend_id, command: trimmed },
          {
            signal: controller.signal,
            onEvent: (event: TerminalEvent) => {
              if (event.type === "start") {
                lastArgv = event.argv
                lastCwd = event.cwd
                appendLine({
                  kind: "info",
                  text: `▶ ${event.argv.join(" ")}`,
                })
              } else if (event.type === "stdout" || event.type === "stderr") {
                sawOutput = true
                appendLine({
                  kind: event.type,
                  text: event.data ?? "",
                })
              } else if (event.type === "exit") {
                const ok = event.code === 0
                appendLine({
                  kind: ok ? "info" : "error",
                  text: ok
                    ? "✓ 命令完成 (exit 0)"
                    : `✗ 命令失败 (exit ${event.code})`,
                })
                if (!ok && !sawOutput && lastArgv) {
                  // Help users diagnose silent failures: dump the
                  // resolved argv + cwd so they can spot a missing
                  // executable, broken venv path, etc.
                  appendLine({
                    kind: "error",
                    text: `  argv: ${lastArgv.join(" ")}`,
                  })
                  if (lastCwd) {
                    appendLine({
                      kind: "error",
                      text: `  cwd:  ${lastCwd}`,
                    })
                  }
                  appendLine({
                    kind: "error",
                    text: "  无任何输出。请确认该后端的 venv python 可用、且命令本身在该 venv 中存在。",
                  })
                }
              } else if (event.type === "error") {
                appendLine({ kind: "error", text: event.data ?? "未知错误" })
              }
            },
          },
        )
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          appendLine({ kind: "info", text: "⏹ 已取消" })
        } else {
          appendLine({
            kind: "error",
            text: err instanceof Error ? err.message : String(err),
          })
        }
      } finally {
        setRunning(false)
        if (abortRef.current === controller) abortRef.current = null
      }
    },
    [activeSession, running, appendLine],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => setLines([]), [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-5 w-full max-w-[1400px]">
        <header className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            后端 · venv 维护
          </div>
          <h1 className="text-2xl font-semibold tracking-tight inline-flex items-center gap-2">
            <TerminalIcon className="size-5 text-muted-foreground" />
            终端
          </h1>
          <p className="text-sm text-muted-foreground">
            选择已安装的后端，在它的虚拟环境中跑 pip / uv / python 命令。
            默认仅允许包管理类命令，可在 设置 中开启自由模式。
          </p>
        </header>

        <BackendPicker
          sessions={sessions}
          selected={backendId}
          loading={sessionsQuery.isLoading}
          onChange={setBackendId}
        />

        <UnrestrictedBanner unrestricted={unrestricted} />

        <QuickCommands
          disabled={!activeSession?.ready || running}
          onPick={(cmd) => setInput(cmd)}
          onRun={(cmd) => runCommand(cmd)}
        />

        <PseudoTerminal
          prompt={activeSession?.prompt ?? "$"}
          ready={!!activeSession?.ready}
          running={running}
          lines={lines}
          input={input}
          history={history}
          onInputChange={setInput}
          onSubmit={runCommand}
          onCancel={cancel}
          onClear={clear}
        />
      </div>
    </div>
  )
}

function UnrestrictedBanner({ unrestricted }: { unrestricted: boolean }) {
  if (!unrestricted) return null
  return (
    <div className="rounded-[6px] border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] flex items-start gap-2">
      <ShieldAlert className="size-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
      <div className="space-y-0.5 flex-1 min-w-0">
        <div className="font-medium">自由命令模式已开启</div>
        <div className="text-muted-foreground">
          任意命令都能在所选后端的 venv 内运行。请确认你信任此 LoraHub 实例所在的环境。
        </div>
      </div>
      <Badge variant="outline" className="rounded-[2px] text-[10px]">
        在「设置 → 终端」关闭
      </Badge>
    </div>
  )
}
