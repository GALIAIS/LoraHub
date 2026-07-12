import { useEffect, useMemo, useRef, useState } from "react"
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso"
import type { TrainingEvent } from "@/lib/api"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { ArrowDown, Eraser } from "lucide-react"
import { cn } from "@/lib/utils"
import { parseAnsi, stripAnsi } from "./ansi"
import {
  buildLogLines,
  matchesLogFilter,
  type LogFilter,
  type LogTone,
} from "./terminal-log-model"

const MAX_LINES = 5000
// Persist whether the user prefers the always-dark "terminal" look or the
// theme-following look. Default is theme-following so the panel matches
// the rest of LoraHub's surfaces in both themes.
const STYLE_KEY = "lorahub.jobs.terminalLog.darkMode"

function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  const h = String(d.getHours()).padStart(2, "0")
  const m = String(d.getMinutes()).padStart(2, "0")
  const s = String(d.getSeconds()).padStart(2, "0")
  return `${h}:${m}:${s}`
}

// Theme-following tone tokens (default look).
//
// Each tone keeps a hue used in both modes; light text uses the 700
// shade so it stays legible on light backgrounds, dark text uses the
// 400 shade so it pops on the muted dark surface.
const TONE_THEMED: Record<LogTone, string> = {
  default: "text-foreground/90",
  debug: "text-muted-foreground/70",
  error: "text-red-700 dark:text-red-400",
  warn: "text-amber-700 dark:text-amber-300",
  progress: "text-cyan-700 dark:text-cyan-400",
  gpu: "text-sky-700 dark:text-sky-400",
  checkpoint: "text-emerald-700 dark:text-emerald-400",
  sample: "text-fuchsia-700 dark:text-fuchsia-400",
  epoch: "text-violet-700 dark:text-violet-400",
  done: "text-emerald-700 dark:text-emerald-400",
}

// Always-dark "terminal" look — keeps the original 400-shade palette
// against a pinned zinc-950 background, regardless of the current theme.
const TONE_TERMINAL: Record<LogTone, string> = {
  default: "text-zinc-100",
  debug: "text-zinc-500",
  error: "text-red-400",
  warn: "text-amber-300",
  progress: "text-cyan-400",
  gpu: "text-sky-400",
  checkpoint: "text-emerald-400",
  sample: "text-fuchsia-400",
  epoch: "text-blue-300",
  done: "text-emerald-300",
}

const BORDER_TONE: Record<LogTone, string> = {
  default: "border-l-transparent",
  debug: "border-l-zinc-400/30",
  error: "border-l-red-500/90",
  warn: "border-l-amber-500/80",
  progress: "border-l-cyan-500/70",
  gpu: "border-l-sky-500/60",
  checkpoint: "border-l-emerald-500/80",
  sample: "border-l-fuchsia-500/80",
  epoch: "border-l-violet-500/70",
  done: "border-l-emerald-500/80",
}

const ROW_TONE: Record<LogTone, string> = {
  default: "",
  debug: "opacity-80",
  error: "bg-red-500/[0.045]",
  warn: "bg-amber-500/[0.04]",
  progress: "bg-cyan-500/[0.035]",
  gpu: "bg-sky-500/[0.03]",
  checkpoint: "bg-emerald-500/[0.035]",
  sample: "bg-fuchsia-500/[0.035]",
  epoch: "bg-violet-500/[0.035]",
  done: "bg-emerald-500/[0.04]",
}

const FILTERS: Array<{ value: LogFilter; label: string }> = [
  { value: "training", label: "训练" },
  { value: "all", label: "全部" },
  { value: "warning", label: "警告" },
  { value: "error", label: "错误" },
  { value: "gpu", label: "GPU" },
]

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
  const [filter, setFilter] = useState<LogFilter>("training")
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
  const virtuosoRef = useRef<VirtuosoHandle | null>(null)
  const [atBottom, setAtBottom] = useState(true)

  const visibleEvents = useMemo(() => {
    if (clearAfter <= 0) return events
    return events.slice(clearAfter)
  }, [events, clearAfter])

  const lines = useMemo(() => {
    const sliced =
      visibleEvents.length > MAX_LINES
        ? visibleEvents.slice(visibleEvents.length - MAX_LINES)
        : visibleEvents
    return buildLogLines(sliced, fallbackTotalSteps)
  }, [visibleEvents, fallbackTotalSteps])

  const filteredLines = useMemo(() => {
    const q = query.toLowerCase()
    return lines.filter(
      (line) =>
        matchesLogFilter(line, filter) &&
        (!q ||
          stripAnsi(`${line.message} ${line.source ?? ""}`)
            .toLowerCase()
            .includes(q)),
    )
  }, [filter, lines, query])

  function clearScreen() {
    setClearAfter(events.length)
    setAtBottom(true)
  }

  function scrollToLatest() {
    if (filteredLines.length === 0) return
    virtuosoRef.current?.scrollToIndex({
      index: filteredLines.length - 1,
      align: "end",
      behavior: "auto",
    })
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
    ? "text-zinc-400 font-semibold tracking-wide"
    : "text-muted-foreground/80 font-semibold tracking-wide"
  const sourceClass = darkMode
    ? "text-zinc-600"
    : "text-muted-foreground/55"
  const bodyTextDefault = darkMode
    ? "font-mono text-[12px] leading-[1.5] text-zinc-100"
    : "font-mono text-[12px] leading-[1.5] text-foreground/90"
  const placeholderClass = darkMode
    ? "px-3 py-6 text-center text-zinc-500"
    : "px-3 py-6 text-center text-muted-foreground"

  return (
    <div
      className={cn(
        "relative h-full flex-1 min-h-0 flex flex-col rounded-[6px] border overflow-hidden",
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
        <div
          className={cn(
            "inline-flex h-7 items-center rounded-[4px] border p-[2px]",
            darkMode ? "border-zinc-700 bg-zinc-900/70" : "border-border/60 bg-background/70",
          )}
          aria-label="日志筛选"
        >
          {FILTERS.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => setFilter(item.value)}
              className={cn(
                "h-5 px-2 text-[10px] font-medium transition-colors",
                filter === item.value
                  ? darkMode
                    ? "bg-zinc-700 text-zinc-50"
                    : "bg-primary/12 text-primary"
                  : toolbarMutedClass,
              )}
              aria-pressed={filter === item.value}
            >
              {item.label}
            </button>
          ))}
        </div>
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
              if (v) {
                setAtBottom(true)
                requestAnimationFrame(scrollToLatest)
              }
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
        className={cn(
          "flex-1 min-h-0 flex flex-col",
          bodyTextDefault,
        )}
      >
        {filteredLines.length === 0 ? (
          <div className={placeholderClass}>
            {events.length === 0
              ? "尚未收到任何日志…"
              : query
                ? "没有匹配的行。"
                : "屏幕已清空，等待新日志…"}
          </div>
        ) : (
          <Virtuoso
            ref={virtuosoRef}
            data={filteredLines}
            className="min-h-0 flex-1"
            atBottomThreshold={8}
            atBottomStateChange={setAtBottom}
            followOutput={(isAtBottom) =>
              autoScroll && isAtBottom ? "auto" : false
            }
            computeItemKey={(_, line) => line.key}
            increaseViewportBy={240}
            itemContent={(_, line) => {
              const chunks = parseAnsi(line.message)
              const toneClass = tonePalette[line.tone]
              return (
                <div
                  key={line.key}
                  className={cn(
                    "grid grid-cols-[64px_54px_minmax(0,1fr)] items-start gap-x-2 border-l-2 px-3 py-[3px] lg:grid-cols-[72px_64px_minmax(0,1fr)_minmax(72px,auto)]",
                    BORDER_TONE[line.tone],
                    ROW_TONE[line.tone],
                    rowHoverClass,
                  )}
                >
                  <span className={timeClass}>[{formatTime(line.ts)}]</span>
                  <span
                    className={cn(
                      levelClass,
                      line.tone === "default" ? "" : toneClass,
                    )}
                  >
                    {line.level}
                  </span>
                  <span
                    className={cn(
                      "min-w-0 whitespace-pre-wrap break-words",
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
                  <span
                    className={cn(
                      "hidden max-w-[180px] truncate text-right text-[10px] lg:block",
                      sourceClass,
                    )}
                    title={line.source ?? line.stream ?? undefined}
                  >
                    {line.source ?? line.stream ?? ""}
                  </span>
                </div>
              )
            }}
          />
        )}
      </div>
      {!atBottom && filteredLines.length > 0 ? (
        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={() => {
            setAtBottom(true)
            scrollToLatest()
          }}
          className={cn(
            "absolute bottom-4 right-5 z-10 size-8 rounded-full",
            darkMode && "border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800",
          )}
          title="跳到最新日志"
          aria-label="跳到最新日志"
        >
          <ArrowDown className="size-3.5" />
        </Button>
      ) : null}
    </div>
  )
}
