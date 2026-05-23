import { useEffect, useMemo, useRef, useState } from "react"
import { Plus, X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * Tag chip editor for comma-separated captions. Each tag is rendered
 * as a removable chip; typing in the trailing input + Enter / comma /
 * blur commits a new tag. Order is preserved so users can drag (future)
 * or just visually rearrange the prompt the way training pipelines
 * actually consume it.
 *
 * The component is fully controlled — `value` is the raw comma-joined
 * string and `onChange` fires with the next string. We dedupe on commit
 * (same tag added twice is a no-op) but never reorder behind the user.
 */
export interface TagChipEditorProps {
  value: string
  onChange: (next: string) => void
  /**
   * Save the edits, e.g. enqueue a replace_caption op. Optional.
   *
   * Receives the just-committed value as an argument because React state
   * updates are async — `onSave` cannot read the latest `captionDraft`
   * from the parent's closure if the user hits Save without first
   * pressing Enter / comma to flush the trailing input. Always trust
   * the argument over any captured prop.
   */
  onSave?: (committed: string) => void
  /** Cancel editing without committing — drops local draft state. */
  onCancel?: () => void
  /** Inline-mode: hide the explicit Save / Cancel buttons. */
  compact?: boolean
  /** Disable interaction while a save is in flight. */
  disabled?: boolean
}

function splitTags(raw: string): string[] {
  return raw
    .split(/[,，]/g)
    .map((t) => t.trim())
    .filter(Boolean)
}

export function TagChipEditor({
  value,
  onChange,
  onSave,
  onCancel,
  compact = false,
  disabled,
}: TagChipEditorProps) {
  const [draftInput, setDraftInput] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  const tags = useMemo(() => splitTags(value), [value])

  // Re-emit a normalized join so trailing whitespace / stray commas
  // stay out of the persisted caption file. Returns the committed
  // string so callers (e.g. addCurrentDraft) can hand it straight to
  // onSave without waiting for the async setState round-trip.
  const commit = (nextTags: string[]): string => {
    const seen = new Set<string>()
    const deduped: string[] = []
    for (const t of nextTags) {
      const norm = t.trim()
      if (!norm) continue
      if (seen.has(norm)) continue
      seen.add(norm)
      deduped.push(norm)
    }
    const joined = deduped.join(", ")
    onChange(joined)
    return joined
  }

  // Returns the freshly-committed value so onSave can flush trailing
  // input without depending on React state having already propagated.
  const addCurrentDraft = (): string => {
    const candidates = splitTags(draftInput)
    if (candidates.length === 0) {
      setDraftInput("")
      return value
    }
    const next = commit([...tags, ...candidates])
    setDraftInput("")
    return next
  }

  const removeAt = (index: number) => {
    const next = tags.slice()
    next.splice(index, 1)
    commit(next)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault()
      addCurrentDraft()
    } else if (e.key === "," || e.key === "，") {
      e.preventDefault()
      addCurrentDraft()
    } else if (e.key === "Backspace" && draftInput === "" && tags.length > 0) {
      // Backspace on empty input pops the last chip — same gesture as
      // Slack / GitHub label chips.
      e.preventDefault()
      removeAt(tags.length - 1)
    }
  }

  // Re-focus the input whenever the underlying caption changes from
  // outside (e.g. AI re-tag), so the user can keep typing.
  useEffect(() => {
    if (!compact && inputRef.current) {
      inputRef.current.focus()
    }
  }, [compact])

  return (
    <div
      className={cn(
        "rounded-[4px] border bg-background px-2 py-1.5",
        disabled && "opacity-60 pointer-events-none",
      )}
    >
      <div className="flex flex-wrap items-center gap-1">
        {tags.map((t, i) => (
          <Badge
            key={`${t}-${i}`}
            variant="secondary"
            className="rounded-[2px] gap-1 text-[10.5px] font-mono py-0.5 pr-1"
          >
            <span className="truncate max-w-[12rem]" title={t}>
              {t}
            </span>
            <button
              type="button"
              onClick={() => removeAt(i)}
              className="rounded hover:bg-foreground/10 p-0.5"
              aria-label={`移除 ${t}`}
              disabled={disabled}
            >
              <X className="size-2.5" />
            </button>
          </Badge>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={draftInput}
          onChange={(e) => setDraftInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => addCurrentDraft()}
          placeholder={tags.length === 0 ? "输入标签，回车或逗号分隔" : "+ 标签"}
          className="flex-1 min-w-[6rem] bg-transparent text-[12px] font-mono outline-none placeholder:text-muted-foreground/70 py-0.5"
          disabled={disabled}
        />
      </div>
      {!compact && (
        <div className="flex items-center justify-between gap-2 mt-2 pt-2 border-t border-border/50">
          <span className="text-[10px] text-muted-foreground">
            {tags.length} 个标签 · 回车 / 逗号 提交
          </span>
          <div className="flex gap-1.5">
            {onCancel && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onCancel}
                className="h-7 text-[11px]"
                disabled={disabled}
              >
                取消
              </Button>
            )}
            {onSave && (
              <Button
                size="sm"
                onClick={() => {
                  // Flush any pending input first, then hand the
                  // committed string to onSave directly. We do NOT
                  // rely on the parent re-rendering because state
                  // updates are async and onSave's closure would still
                  // see the stale captionDraft.
                  const committed = addCurrentDraft()
                  onSave(committed)
                }}
                className="h-7 text-[11px]"
                disabled={disabled}
              >
                <Plus className="size-3" />
                保存
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
