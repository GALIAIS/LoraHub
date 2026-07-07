import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { BookOpen } from "lucide-react"
import {
  libraryListPrompts,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { PromptLibraryPanel } from "./library/prompt-library-panel"

export type CaptionPromptMode = "style" | "character" | "general"
export type CaptionPromptValue = CaptionPromptMode | `custom:${string}`

export interface CaptionPromptSelection {
  value: CaptionPromptValue
  captionMode: CaptionPromptMode
  promptTemplate?: string
}

const DEFAULTS: { value: CaptionPromptMode; label: string }[] = [
  { value: "style", label: "风格 LoRA（描述 + 修正后的标签，禁画风词）" },
  { value: "character", label: "角色 LoRA（描述 + 修正后的标签，禁外貌词）" },
  { value: "general", label: "通用（描述全部内容）" },
]

function modeFromCategory(category: string): CaptionPromptMode | null {
  if (!category.startsWith("caption:")) return null
  const mode = category.slice("caption:".length)
  return mode === "style" || mode === "character" || mode === "general"
    ? mode
    : null
}

export function CaptionPromptPicker({
  value,
  onChange,
  className,
}: {
  value: CaptionPromptValue
  onChange: (next: CaptionPromptSelection) => void
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const promptsQuery = useQuery({
    queryKey: ["library", "prompts"],
    queryFn: () => libraryListPrompts(),
  })

  const prompts = useMemo(
    () =>
      (promptsQuery.data?.prompts ?? []).filter((p) =>
        Boolean(modeFromCategory(p.category)),
      ),
    [promptsQuery.data?.prompts],
  )

  useEffect(() => {
    if (!value.startsWith("custom:") || promptsQuery.isLoading) return
    const id = value.slice("custom:".length)
    if (!prompts.some((p) => p.id === id)) {
      onChange({ value: "style", captionMode: "style" })
    }
  }, [onChange, prompts, promptsQuery.isLoading, value])

  const selectValue = (next: string) => {
    if (next === "style" || next === "character" || next === "general") {
      onChange({ value: next, captionMode: next })
      return
    }
    const id = next.replace(/^custom:/, "")
    const prompt = prompts.find((p) => p.id === id)
    if (!prompt) return
    onChange({
      value: `custom:${prompt.id}`,
      captionMode: modeFromCategory(prompt.category) ?? "style",
      promptTemplate: prompt.body,
    })
  }

  return (
    <>
      <div className={["flex gap-2", className].filter(Boolean).join(" ")}>
        <select
          value={value}
          onChange={(e) => selectValue(e.target.value)}
          className="h-8 min-w-0 flex-1 rounded border bg-background px-2 text-xs"
        >
          {DEFAULTS.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
          {prompts.map((p) => (
            <option key={p.id} value={`custom:${p.id}`}>
              {p.name}
            </option>
          ))}
        </select>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 shrink-0 gap-1 px-2 text-xs"
          onClick={() => setOpen(true)}
        >
          <BookOpen className="size-3.5" />
          提示词库
        </Button>
      </div>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="flex h-[min(80vh,44rem)] max-w-[min(calc(100%-2rem),52rem)] flex-col gap-0 p-0">
          <DialogHeader className="shrink-0 border-b px-6 py-5">
            <DialogTitle>Caption 提示词库</DialogTitle>
          </DialogHeader>
          <div className="min-h-0 flex-1">
            <PromptLibraryPanel
              categoryPrefix="caption:"
              defaultDraft={{
                category: "caption:style",
                vars: "tags, wd14_tags, trigger",
              }}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
