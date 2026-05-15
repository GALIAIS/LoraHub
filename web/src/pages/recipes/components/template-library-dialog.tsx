import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { FilePlus2 } from "lucide-react"
import { api, type RecipeTemplate } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export function TemplateLibraryDialog({
  open,
  onOpenChange,
  onUseBlank,
  onCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Called when the user picks the blank template — caller routes to the empty editor. */
  onUseBlank: () => void
  /** Called after a non-blank template was saved as a new recipe. */
  onCreated: (name: string) => void
}) {
  const qc = useQueryClient()
  const templates = useQuery({
    queryKey: ["recipe-templates"],
    queryFn: api.listRecipeTemplates,
    enabled: open,
  })

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Reset internal state on every open so the dialog feels fresh.
  useEffect(() => {
    if (!open) {
      setSelectedId(null)
      setName("")
      setErrorMsg(null)
    }
  }, [open])

  const list = templates.data?.templates ?? []
  const selected = list.find((t) => t.id === selectedId) ?? null

  // When the user picks a template, default the name to "<id>_v1".
  useEffect(() => {
    if (selected) {
      setName(`${selected.id}_v1`)
      setErrorMsg(null)
    }
  }, [selected])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("请先选择模板")
      const trimmed = name.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.saveRecipe(trimmed, selected.recipe, false)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["recipes"] })
      onOpenChange(false)
      onCreated(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  const handleConfirm = () => {
    if (!selected) return
    if (selected.id === "blank") {
      onOpenChange(false)
      onUseBlank()
      return
    }
    mutation.mutate()
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),48rem)]">
        <DialogHeader>
          <DialogTitle>模板库</DialogTitle>
          <DialogDescription>
            从模板创建新配方，或选择「Blank」从空白表单开始。
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[50vh]">
          {templates.isLoading && (
            <div className="text-sm text-muted-foreground px-1 py-6 text-center">加载中…</div>
          )}
          {templates.isError && (
            <div className="text-sm text-destructive px-1 py-6 text-center">
              模板加载失败：{(templates.error as Error).message}
            </div>
          )}
          {!templates.isLoading && list.length > 0 && (
            <div className="grid grid-cols-2 gap-3 p-1">
              {list.map((tpl) => (
                <TemplateCard
                  key={tpl.id}
                  template={tpl}
                  active={selectedId === tpl.id}
                  onSelect={() => setSelectedId(tpl.id)}
                />
              ))}
            </div>
          )}
        </ScrollArea>

        {selected && selected.id !== "blank" && (
          <div className="flex flex-col gap-1.5">
            <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              新配方名称
            </Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="font-mono"
              placeholder="my_recipe_v1"
            />
          </div>
        )}

        {errorMsg && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMsg}
          </div>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            取消
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={
              !selected ||
              mutation.isPending ||
              (selected.id !== "blank" && !name.trim())
            }
          >
            <FilePlus2 className="size-3" />
            {mutation.isPending
              ? "创建中…"
              : selected?.id === "blank"
                ? "进入空白表单"
                : "创建配方"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function TemplateCard({
  template,
  active,
  onSelect,
}: {
  template: RecipeTemplate
  active: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "text-left rounded-[6px] border px-4 py-3 transition-colors shadow-[var(--panel-shadow)]",
        active
          ? "border-primary bg-accent/60"
          : "border-border/60 bg-background/60 hover:bg-muted/40",
      )}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-sm font-medium truncate">{template.name}</span>
        <Badge
          variant="outline"
          className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
        >
          {template.arch}
        </Badge>
      </div>
      <div className="text-xs text-muted-foreground leading-relaxed">
        {template.description}
      </div>
    </button>
  )
}
