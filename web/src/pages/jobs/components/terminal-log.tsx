import { useEffect, useMemo, useRef, useState } from "react"
import type { TrainingEvent } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Eraser } from "lucide-react"
import { cn } from "@/lib/utils"
import { parseAnsi, stripAnsi } from "./ansi"

const MAX_LINES = 5000
// Pixel slack for "is the user already at the bottom?" — anything within this
// distance counts as following the tail.
const STICK_TO_BOTTOM_PX = 4
// Persist whether the user prefers the always-dark "terminal" look or the
// theme-following look. Default is theme-following so the panel matches
// the rest of LoraHub's surfaces in both themes.
const STYLE_KEY = "lorahub.jobs.terminalLog.darkMode"

interface LogLine {
  ts: number
  level: string
  message: string
  // Tone tokens stay theme-agnostic. We pick paired light/dark utilities
  // (e.g. text-X-700 dark:text-X-400) so the same line renders legibly
  // on both surfaces; the always-dark "terminal" look maps the same
  // tokens to their dark variant via a wrapper class.
  tone: LogTone
  borderClass: string
  // Original event index lets us key without identity drift on re-renders.
  key: string
}

type LogTone =
  | "default"
  | "error"
  | "warn"
  | "step"
  | "checkpoint"
  | "sample"
  | "epoch"
  | "done"

function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  const h = String(d.getHours()).padStart(2, "0")
  const m = String(d.getMinutes()).padStart(2, "0")
  const s = String(d.getSeconds()).padStart(2, "0")
  return `${h}:${m}:${s}`
}

function eventToLine(
  event: TrainingEvent,
  index: number,
  fallbackTotalSteps: number | null,
): LogLine {
  const p = event.payload
  const stepTotal =
    typeof p.total_steps === "number" && p.total_steps > 0
      ? (p.total_steps as number)
      : fallbackTotalSteps
  const rawMessage =
    typeof p.message === "string"
      ? (p.message as string)
      : event.type === "step"
        ? `第 ${p.step}/${stepTotal ?? "?"} 步${
            typeof p.loss === "number"
              ? ` · loss=${(p.loss as number).toFixed(4)}`
              : ""
          }`
        : event.type === "epoch_end"
          ? `第 ${p.epoch}/${p.total_epochs ?? "?"} 回合结束`
          : event.type === "checkpoint_saved"
            ? `保存检查点：${p.path ?? ""}`
            : event.type === "sample_ready"
              ? `生成样本：${p.path ?? ""}`
              : event.type === "done"
                ? `已结束 · 返回码=${p.returncode ?? "?"}`
                : event.type === "start"
                  ? "训练已启动"
                  : event.type === "cancel"
                    ? "已请求取消"
                    : JSON.stringify(p)

  // Resolve the level. We respect explicit `payload.level` (Python logging uses
  // this for forwarded log lines), otherwise derive from event type or content.
  const explicitLevel =
    typeof p.level === "string" ? (p.level as string).toUpperCase() : null
  let level = explicitLevel ?? event.type.toUpperCase()
  // Cancel-shaped messages (Ctrl-C, sigkill_handler, deepspeed launch's
  // `exits with return code = -2`) must NOT render red — they're a clean
  // user stop, not a failure. We override `looksLikeError` for them.
  const looksLikeCancel =
    /\b(?:keyboardinterrupt|killing subprocess|exits with return code = -(?:2|9|15))\b/i.test(
      rawMessage,
    )
  // `Traceback (most recent call last):` is only a banner — the exception
  // summary that follows is the real error signal, and that summary has
  // its own `XxxError`/`XxxException` keyword which matches below.
  const looksLikeError =
    !looksLikeCancel &&
    (event.type === "error" ||
      /\b(error|fail(ed|ure)?|fatal|exception)\b/i.test(rawMessage))

  let toneVal: LogTone = "default"
  let borderClass = "border-l-transparent"
  if (looksLikeError || level === "ERROR" || level === "CRITICAL") {
    toneVal = "error"
    borderClass = "border-l-red-500/80"
    if (level !== "ERROR" && level !== "CRITICAL") level = "ERROR"
  } else if (event.type === "step") {
    toneVal = "step"
  } else if (event.type === "checkpoint_saved") {
    toneVal = "checkpoint"
  } else if (event.type === "sample_ready") {
    toneVal = "sample"
  } else if (event.type === "done") {
    toneVal = "done"
  } else if (event.type === "epoch_end") {
    toneVal = "epoch"
  } else if (level === "WARNING" || level === "WARN") {
    toneVal = "warn"
  }

  return {
    ts: event.timestamp,
    level,
    message: rawMessage,
    tone: toneVal,
    borderClass,
    key: `${index}-${event.timestamp}-${event.type}`,
  }
}

// Theme-following tone tokens (default look).
//
// Each tone keeps a hue used in both modes; light text uses the 700
// shade so it stays legible on light backgrounds, dark text uses the
// 400 shade so it pops on the muted dark surface.
const TONE_THEMED: Record<LogTone, string> = {
  default: "text-foreground/90",
  error: "text-red-700 dark:text-red-400",
  warn: "text-amber-700 dark:text-amber-300",
  step: "text-cyan-700 dark:text-cyan-400",
  checkpoint: "text-emerald-700 dark:text-emerald-400",
  sample: "text-fuchsia-700 dark:text-fuchsia-400",
  epoch: "text-violet-700 dark:text-violet-400",
  done: "text-emerald-700 dark:text-emerald-400",
}

// Always-dark "terminal" look — keeps the original 400-shade palette
// against a pinned zinc-950 background, regardless of the current theme.
const TONE_TERMINAL: Record<LogTone, string> = {
  default: "text-zinc-100",
  error: "text-red-400",
  warn: "text-amber-300",
  step: "text-cyan-400",
  checkpoint: "text-emerald-400",
  sample: "text-fuchsia-400",
  epoch: "text-blue-300",
  done: "text-emerald-300",
}

function highlightChunk(
  text: string,
  baseClass: string,
  needle: string,
  startKey: string,
): React.ReactNode[] {
  if (!needle) {
    return [
      <span key={startKey} className={baseClass}>
        {text}
      </span>,
    ]
  }
  const lower = text.toLowerCase()
  const lowerNeedle = needle.toLowerCase()
  const out: React.ReactNode[] = []
  let cursor = 0
  let n = 0
  while (cursor < text.length) {
    const idx = lower.indexOf(lowerNeedle, cursor)
    if (idx === -1) {
      out.push(
        <span key={`${startKey}-${n++}`} className={baseClass}>
          {text.slice(cursor)}
        </span>,
      )
      break
    }
    if (idx > cursor) {
      out.push(
        <span key={`${startKey}-${n++}`} className={baseClass}>
          {text.slice(cursor, idx)}
        </span>,
      )
    }
    out.push(
      <mark
        key={`${startKey}-${n++}`}
        className={cn(baseClass, "bg-yellow-500/30 text-inherit rounded-[1px]")}
      >
        {text.slice(idx, idx + needle.length)}
      </mark>,
    )
    cursor = idx + needle.length
  }
  return out
}

export function TerminalLog({
  events,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  fallbackTotalSteps?: number | null
}) {
  const [query, setQuery] = useState("")
  const [autoScroll, setAutoScroll] = useState(true)
  const [clearAfter, setClearAfter] = useState(0)
  // `darkMode=true` pins the always-dark terminal look; `false` follows
  // whatever theme LoraHub is in. Defaults to following the theme so the
  // panel matches the rest of the workbench out of the box.
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    if (typeof window === "undefined") return false
    return window.localStorage.getItem(STYLE_KEY) === "1"
  })
  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(STYLE_KEY, darkMode ? "1" : "0")
  }, [darkMode])
  const scrollRef = useRef<HTMLDivElement | null>(null)
  // True while the user has actively scrolled away from the tail. Auto-follow
  // pauses while this is set; the Switch state above tracks the user's intent.
  const [tailing, setTailing] = useState(true)

  const visibleEvents = useMemo(() => {
    if (clearAfter <= 0) return events
    return events.slice(clearAfter)
  }, [events, clearAfter])

  const lines = useMemo(() => {
    const sliced =
      visibleEvents.length > MAX_LINES
        ? visibleEvents.slice(visibleEvents.length - MAX_LINES)
        : visibleEvents
    return sliced.map((e, i) => eventToLine(e, i, fallbackTotalSteps))
  }, [visibleEvents, fallbackTotalSteps])

  const filteredLines = useMemo(() => {
    if (!query) return lines
    const q = query.toLowerCase()
    return lines.filter((l) => stripAnsi(l.message).toLowerCase().includes(q))
  }, [lines, query])

  // Track the user's scroll position so we know whether to keep snapping the
  // viewport to the bottom on new lines. Switch state is the user's hard
  // preference; tailing is the runtime guard.
  function onScroll() {
    const node = scrollRef.current
    if (!node) return
    const distanceFromBottom =
      node.scrollHeight - node.scrollTop - node.clientHeight
    const atBottom = distanceFromBottom <= STICK_TO_BOTTOM_PX
    setTailing(atBottom)
  }

  useEffect(() => {
    if (!autoScroll || !tailing) return
    const node = scrollRef.current
    if (!node) return
    node.scrollTop = node.scrollHeight
  }, [filteredLines, autoScroll, tailing])

  function clearScreen() {
    setClearAfter(events.length)
    setTailing(true)
  }

  // Theme palette switch — `darkMode` pins a zinc-950 surface, otherwise
  // the panel follows the workbench card theme tokens. Each surface picks
  // a matching tone map so colour contrast stays balanced.
  const tonePalette = darkMode ? TONE_TERMINAL : TONE_THEMED
  const surfaceClass = darkMode
    ? "bg-zinc-950 border-zinc-800"
    : "bg-card text-foreground border-border/40"
  const toolbarClass = darkMode
    ? "bg-zinc-900/60 border-zinc-800/80"
    : "bg-muted/40 border-border/40"
  const toolbarTextClass = darkMode ? "text-zinc-300" : "text-foreground/80"
  const toolbarMutedClass = darkMode ? "text-zinc-500" : "text-muted-foreground"
  const toolbarInputClass = darkMode
    ? "h-7 max-w-[260px] bg-zinc-900/70 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
    : "h-7 max-w-[260px]"
  const toolbarButtonClass = darkMode
    ? "h-7 px-2 bg-zinc-900/70 border-zinc-700 text-zinc-200 hover:bg-zinc-800 hover:text-zinc-50"
    : "h-7 px-2"
  const rowHoverClass = darkMode ? "hover:bg-zinc-800/50" : "hover:bg-muted/40"
  const timeClass = darkMode
    ? "text-zinc-500 shrink-0 tabular-nums"
    : "text-muted-foreground/70 shrink-0 tabular-nums"
  const levelClass = darkMode
    ? "text-zinc-400 shrink-0 w-[72px] tracking-wide"
    : "text-muted-foreground/80 shrink-0 w-[72px] tracking-wide"
  const bodyTextDefault = darkMode
    ? "font-mono text-[12px] leading-[1.5] text-zinc-100"
    : "font-mono text-[12px] leading-[1.5] text-foreground/90"
  const placeholderClass = darkMode
    ? "px-3 py-6 text-center text-zinc-500"
    : "px-3 py-6 text-center text-muted-foreground"

  return (
    <div
      className={cn(
        "flex-1 min-h-0 flex flex-col rounded-[6px] border overflow-hidden",
        surfaceClass,
      )}
    >
      <div
        className={cn(
          "px-3 py-2 border-b flex items-center gap-3 flex-wrap",
          toolbarClass,
        )}
      >
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索日志…"
          className={toolbarInputClass}
        />
        <label
          className={cn(
            "flex items-center gap-2 text-[11px] select-none cursor-pointer",
            toolbarTextClass,
          )}
        >
          <Switch
            size="sm"
            checked={autoScroll}
            onCheckedChange={(v) => {
              setAutoScroll(v)
              if (v) setTailing(true)
            }}
          />
          自动滚到底
        </label>
        <label
          className={cn(
            "flex items-center gap-2 text-[11px] select-none cursor-pointer",
            toolbarTextClass,
          )}
          title="开启后强制使用经典暗色终端外观；关闭则跟随 LoraHub 主题"
        >
          <Switch
            size="sm"
            checked={darkMode}
            onCheckedChange={setDarkMode}
          />
          终端外观
        </label>
        <Button
          variant="outline"
          size="sm"
          onClick={clearScreen}
          className={toolbarButtonClass}
        >
          <Eraser className="size-3" /> 清屏
        </Button>
        <span
          className={cn(
            "ml-auto text-[10px] uppercase tracking-[0.18em]",
            toolbarMutedClass,
          )}
        >
          显示 {filteredLines.length}
          {query ? ` / ${lines.length}` : ""} 行
          {events.length > MAX_LINES + clearAfter
            ? `（共 ${events.length}，已截断到最近 ${MAX_LINES}）`
            : ""}
        </span>
      </div>
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={cn("flex-1 min-h-0 overflow-y-auto", bodyTextDefault)}
      >
        {filteredLines.length === 0 ? (
          <div className={placeholderClass}>
            {events.length === 0
              ? "尚未收到任何日志……"
              : query
                ? "没有匹配的行。"
                : "屏幕已清空，等待新日志……"}
          </div>
        ) : (
          <ul>
            {filteredLines.map((line) => {
              const chunks = parseAnsi(line.message)
              const toneClass = tonePalette[line.tone]
              return (
                <li
                  key={line.key}
                  className={cn(
                    "px-3 py-[2px] flex gap-2 items-baseline border-l-2",
                    line.borderClass,
                    rowHoverClass,
                  )}
                >
                  <span className={timeClass}>[{formatTime(line.ts)}]</span>
                  <span className={levelClass}>{line.level}</span>
                  <span
                    className={cn(
                      "whitespace-pre-wrap break-all flex-1",
                      toneClass,
                    )}
                  >
                    {chunks.length === 0
                      ? highlightChunk(
                          line.message,
                          "",
                          query,
                          `${line.key}-plain`,
                        )
                      : chunks.flatMap((c, ci) =>
                          highlightChunk(
                            c.text,
                            c.className,
                            query,
                            `${line.key}-${ci}`,
                          ),
                        )}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}
