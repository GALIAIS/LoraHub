/**
 * PseudoTerminal — scrollback view + bottom input row.
 *
 * Implementation notes:
 *
 * - We auto-scroll to the bottom on every new line, but bail out if the
 *   user has scrolled up. "Scrolled up" is checked via the distance of
 *   the scrollTop+clientHeight from scrollHeight; staying within 32px is
 *   considered "still pinned to bottom".
 * - History navigation matches a real shell: ↑ walks backwards through
 *   the persisted command list, ↓ walks forward and clears once past
 *   the latest. We snapshot the in-progress draft when the user first
 *   presses ↑, so they don't lose half-typed input.
 * - Ctrl+L clears the scrollback (matches bash). Ctrl+C cancels the
 *   running command without clearing the buffer.
 */
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react"
import { Loader2, Square, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { TerminalLine } from "./types"

interface PseudoTerminalProps {
  prompt: string
  ready: boolean
  running: boolean
  lines: TerminalLine[]
  input: string
  history: string[]
  onInputChange: (next: string) => void
  onSubmit: (command: string) => void
  onCancel: () => void
  onClear: () => void
}

export function PseudoTerminal({
  prompt,
  ready,
  running,
  lines,
  input,
  history,
  onInputChange,
  onSubmit,
  onCancel,
  onClear,
}: PseudoTerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [historyCursor, setHistoryCursor] = useState<number | null>(null)
  const [draftSnapshot, setDraftSnapshot] = useState<string | null>(null)

  // Reset history cursor whenever the user types (anything that isn't
  // ↑/↓ navigation). Without this, after walking up to "pip install
  // foo" + editing it, ↑ would still treat the new text as a history
  // entry and clobber the edit.
  const onChange = (next: string) => {
    setHistoryCursor(null)
    setDraftSnapshot(null)
    onInputChange(next)
  }

  const submit = () => {
    if (!input.trim()) return
    setHistoryCursor(null)
    setDraftSnapshot(null)
    onSubmit(input)
  }

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault()
      submit()
      return
    }
    if (e.ctrlKey && (e.key === "l" || e.key === "L")) {
      e.preventDefault()
      onClear()
      return
    }
    if (e.ctrlKey && (e.key === "c" || e.key === "C")) {
      // Only intercept when something is running and there's no text
      // selection — Ctrl+C with selected text should still copy.
      const selection = window.getSelection()?.toString() ?? ""
      if (running && !selection) {
        e.preventDefault()
        onCancel()
      }
      return
    }
    if (e.key === "ArrowUp") {
      if (history.length === 0) return
      e.preventDefault()
      if (historyCursor === null) {
        setDraftSnapshot(input)
        const idx = history.length - 1
        setHistoryCursor(idx)
        onInputChange(history[idx])
      } else if (historyCursor > 0) {
        const idx = historyCursor - 1
        setHistoryCursor(idx)
        onInputChange(history[idx])
      }
      return
    }
    if (e.key === "ArrowDown") {
      if (historyCursor === null) return
      e.preventDefault()
      if (historyCursor >= history.length - 1) {
        setHistoryCursor(null)
        onInputChange(draftSnapshot ?? "")
        setDraftSnapshot(null)
      } else {
        const idx = historyCursor + 1
        setHistoryCursor(idx)
        onInputChange(history[idx])
      }
      return
    }
  }

  // Auto-scroll on new lines unless the user has scrolled up to read
  // history. ~32px tolerance covers the common case where the layout
  // hasn't fully settled yet.
  const pinnedToBottomRef = useRef(true)
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const dist = el.scrollHeight - el.scrollTop - el.clientHeight
      pinnedToBottomRef.current = dist < 32
    }
    el.addEventListener("scroll", onScroll)
    return () => el.removeEventListener("scroll", onScroll)
  }, [])
  useEffect(() => {
    if (pinnedToBottomRef.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
    }
  }, [lines])

  // Focus input when the page mounts and after a backend swap.
  useEffect(() => {
    if (ready) inputRef.current?.focus()
  }, [ready, prompt])

  const promptClasses = useMemo(
    () =>
      cn(
        "select-none text-[11px] font-mono shrink-0",
        running ? "text-amber-500/80" : "text-emerald-600/80 dark:text-emerald-400/80",
      ),
    [running],
  )

  return (
    <div className="rounded-[6px] border border-border/60 bg-zinc-950 dark:bg-zinc-950 text-zinc-100 overflow-hidden flex flex-col min-h-[24rem] shadow-[var(--panel-shadow)]">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-zinc-800 bg-zinc-900/60 text-[11px] text-zinc-400">
        <span className="font-mono">{prompt}</span>
        <span className="ml-auto" />
        {running ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCancel}
            className="h-6 text-[11px] text-amber-300 hover:text-amber-200 hover:bg-amber-500/10"
            title="中断当前命令 (Ctrl+C)"
          >
            <Square className="size-3" />
            停止
          </Button>
        ) : null}
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          className="h-6 text-[11px] text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800"
          title="清屏 (Ctrl+L)"
          disabled={lines.length === 0}
        >
          <Trash2 className="size-3" />
          清屏
        </Button>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto px-3 py-2 font-mono text-[12px] leading-relaxed whitespace-pre-wrap break-all"
      >
        {lines.length === 0 ? (
          <div className="text-zinc-500 italic">
            {ready
              ? "在下方输入命令，或点击「快捷命令」开始。"
              : "选择一个已安装的后端后即可使用终端。"}
          </div>
        ) : (
          lines.map((line, idx) => (
            <LineRow key={idx} line={line} />
          ))
        )}
      </div>

      <div className="flex items-center gap-2 px-3 py-2 border-t border-zinc-800 bg-zinc-900/40">
        <span className={promptClasses}>{prompt}</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          placeholder={ready ? "" : "选择一个已安装的后端后再输入命令"}
          className="flex-1 bg-transparent border-0 outline-none font-mono text-[12px] text-zinc-100 placeholder:text-zinc-600"
          spellCheck={false}
          autoComplete="off"
          disabled={!ready}
        />
        {running && (
          <Loader2 className="size-3 animate-spin text-amber-400 shrink-0" />
        )}
      </div>
    </div>
  )
}

function LineRow({ line }: { line: TerminalLine }) {
  if (line.kind === "prompt") {
    return (
      <div className="flex gap-2 items-baseline">
        <span className="text-emerald-400/90 select-none">{line.prompt}</span>
        <span className="text-zinc-100">{line.text}</span>
      </div>
    )
  }
  const className = cn(
    "block",
    line.kind === "stdout" && "text-zinc-200",
    line.kind === "stderr" && "text-amber-300",
    line.kind === "info" && "text-cyan-400/90",
    line.kind === "error" && "text-rose-400",
  )
  return <span className={className}>{line.text}</span>
}
