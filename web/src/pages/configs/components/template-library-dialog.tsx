import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArrowLeft, FilePlus2 } from "lucide-react"
import { api, type BackendId, type ConfigTemplate } from "@/lib/api"
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

type Stage = "pick" | "fill"

export function TemplateLibraryDialog({
  open,
  onOpenChange,
  onUseBlank,
  onCreated,
  activeBackend,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Called when the user picks the blank template — caller routes to the empty editor. */
  onUseBlank: () => void
  /** Called after a non-blank template was saved as a new config. */
  onCreated: (name: string) => void
  activeBackend?: BackendId
}) {
  const qc = useQueryClient()
  const templates = useQuery({
    queryKey: ["config-templates"],
    queryFn: api.listConfigTemplates,
    enabled: open,
  })

  const [stage, setStage] = useState<Stage>("pick")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [values, setValues] = useState<Record<string, string>>({})
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Reset internal state on every open so the dialog feels fresh.
  useEffect(() => {
    if (!open) {
      setStage("pick")
      setSelectedId(null)
      setName("")
      setValues({})
      setErrorMsg(null)
    }
  }, [open])

  const allTemplates = templates.data?.templates ?? []
  const list = useMemo(
    () =>
      activeBackend
        ? allTemplates.filter(
            (tpl) =>
              tpl.id === "blank" || templateBackend(tpl) === activeBackend,
          )
        : allTemplates,
    [activeBackend, allTemplates],
  )
  const selected = useMemo(
    () => list.find((t) => t.id === selectedId) ?? null,
    [list, selectedId],
  )

  const hasPlaceholders = (selected?.placeholders?.length ?? 0) > 0

  // When the user picks a template, default the name to "<id>_v1".
  useEffect(() => {
    if (selected) {
      setName(`${selected.id}_v1`)
      // Seed each placeholder field with its placeholder hint so the user
      // sees concrete examples instead of empty inputs.
      const seeded: Record<string, string> = {}
      for (const ph of selected.placeholders ?? []) {
        seeded[ph.key] = ""
      }
      setValues(seeded)
      setErrorMsg(null)
    }
  }, [selected])

  const saveBlankCopy = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("请先选择模板")
      const trimmed = name.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.saveConfig(trimmed, selected.config, false)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["configs"] })
      onOpenChange(false)
      onCreated(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  const instantiate = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("请先选择模板")
      const trimmed = name.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.instantiateConfigTemplate(selected.id, {
        name: trimmed,
        values,
      })
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["configs"] })
      onOpenChange(false)
      onCreated(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  const pending = instantiate.isPending || saveBlankCopy.isPending

  const handleConfirm = () => {
    if (!selected) return
    if (selected.id === "blank") {
      onOpenChange(false)
      onUseBlank()
      return
    }
    if (stage === "pick") {
      // Templates without placeholders skip the fill stage and write the
      // body verbatim — preserves the old behaviour for stripped-down YAMLs.
      if (!hasPlaceholders) {
        saveBlankCopy.mutate()
        return
      }
      setErrorMsg(null)
      setStage("fill")
      return
    }
    instantiate.mutate()
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),48rem)]">
        <DialogHeader>
          <DialogTitle>
            {stage === "fill" && selected ? `参数化模板：${selected.name}` : "模板库"}
          </DialogTitle>
          <DialogDescription>
            {stage === "fill"
              ? "填入此模板需要的关键路径，提交后会写入新的配置文件。"
              : "从模板创建新配置，或选择「Blank」从空白表单开始。"}
          </DialogDescription>
        </DialogHeader>

        {stage === "pick" ? (
          <PickStage
            templates={templates}
            list={list}
            selectedId={selectedId}
            onSelect={setSelectedId}
            selected={selected}
            name={name}
            onNameChange={setName}
          />
        ) : (
          selected && (
            <FillStage
              template={selected}
              name={name}
              onNameChange={setName}
              values={values}
              onValuesChange={setValues}
            />
          )
        )}

        {errorMsg && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMsg}
          </div>
        )}

        <DialogFooter>
          {stage === "fill" && (
            <Button
              variant="ghost"
              onClick={() => setStage("pick")}
              disabled={pending}
            >
              <ArrowLeft className="size-3" /> 返回
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={pending}
          >
            取消
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={
              !selected ||
              pending ||
              (selected.id !== "blank" && !name.trim())
            }
          >
            <FilePlus2 className="size-3" />
            {pending
              ? "创建中…"
              : selected?.id === "blank"
                ? "进入空白表单"
                : stage === "pick" && hasPlaceholders
                  ? "下一步"
                  : "创建配置"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function templateBackend(template: ConfigTemplate): BackendId {
  const backend = template.config.backend
  if (backend && typeof backend === "object" && "type" in backend) {
    const type = String((backend as { type?: unknown }).type)
    if (
      type === "kohya" ||
      type === "diffusion-pipe" ||
      type === "anima_lora" ||
      type === "ai_toolkit"
    ) {
      return type
    }
  }
  return "kohya"
}

function PickStage({
  templates,
  list,
  selectedId,
  onSelect,
  selected,
  name,
  onNameChange,
}: {
  templates: ReturnType<typeof useQuery<{ templates: ConfigTemplate[] }>>
  list: ConfigTemplate[]
  selectedId: string | null
  onSelect: (id: string) => void
  selected: ConfigTemplate | null
  name: string
  onNameChange: (next: string) => void
}) {
  const showNameInput =
    selected !== null &&
    selected.id !== "blank" &&
    (selected.placeholders?.length ?? 0) === 0
  return (
    <>
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
                onSelect={() => onSelect(tpl.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>

      {showNameInput && (
        <div className="flex flex-col gap-1.5">
          <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            新配置名称
          </Label>
          <Input
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            className="font-mono"
            placeholder="my_config_v1"
          />
        </div>
      )}
    </>
  )
}

function FillStage({
  template,
  name,
  onNameChange,
  values,
  onValuesChange,
}: {
  template: ConfigTemplate
  name: string
  onNameChange: (next: string) => void
  values: Record<string, string>
  onValuesChange: (next: Record<string, string>) => void
}) {
  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-1.5">
        <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
          新配置名称
        </Label>
        <Input
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          className="font-mono"
          placeholder="my_config_v1"
        />
      </div>

      <ScrollArea className="max-h-[40vh]">
        <div className="space-y-3 p-1">
          {template.placeholders.map((ph) => (
            <div key={ph.key} className="flex flex-col gap-1.5">
              <Label className="text-[12px] flex items-center justify-between gap-2">
                <span>{ph.label}</span>
                <code className="font-mono text-[10px] text-muted-foreground">
                  {ph.path_field}
                </code>
              </Label>
              <Input
                value={values[ph.key] ?? ""}
                onChange={(e) =>
                  onValuesChange({ ...values, [ph.key]: e.target.value })
                }
                className="font-mono text-[12px]"
                placeholder={ph.placeholder}
              />
            </div>
          ))}
        </div>
      </ScrollArea>
      <p className="text-[11px] text-muted-foreground">
        留空的字段保持模板默认值。提交时后端会用 TrainingConfig 重新校验。
      </p>
    </div>
  )
}

function TemplateCard({
  template,
  active,
  onSelect,
}: {
  template: ConfigTemplate
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
        {(template.placeholders?.length ?? 0) > 0 && (
          <Badge
            variant="secondary"
            className="rounded-[2px] text-[10px] py-0 px-1.5"
          >
            {template.placeholders.length} 项参数
          </Badge>
        )}
      </div>
      <div className="text-xs text-muted-foreground leading-relaxed">
        {template.description}
      </div>
    </button>
  )
}
