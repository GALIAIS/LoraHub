/**
 * 采样提示词编辑对话框。
 *
 * 用户在主表单点击「编辑提示词」按钮即弹出，提供一行一条提示词的
 * 编辑界面：每行支持独立的 negative / cfg / steps / seed / w×h。
 * 保存时整体写回 ConfigFormValue.sampling.prompts；后端在任务启动
 * 时把这份列表物化成 kohya 风格的 prompts.txt（见
 * lifecycle._materialise_prompts_file），所以前端不依赖外部 .txt
 * 文件，所有内容直接保存在 yaml 中。
 *
 * 种子语义与 SeedInput 保持一致：-1 表示「运行时随机」。
 */
import { memo, useEffect, useState } from "react"
import { Plus, Shuffle, Trash2 } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { SamplingPromptValue } from "../types"

const SEED_MAX = 1_125_899_906_842_624

const EMPTY_PROMPT: SamplingPromptValue = {
  prompt: "",
  negative: null,
  cfg: null,
  steps: null,
  seed: -1,
  width: null,
  height: null,
}

export interface PromptsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initial: SamplingPromptValue[]
  defaultResolution?: [number, number] | undefined
  onSave: (next: SamplingPromptValue[]) => void
}

export const PromptsDialog = memo(function PromptsDialog({
  open,
  onOpenChange,
  initial,
  defaultResolution,
  onSave,
}: PromptsDialogProps) {
  const [rows, setRows] = useState<SamplingPromptValue[]>(initial)

  // Reset working buffer whenever the dialog opens — we never persist
  // mid-edit state across closings, so users get a clean slate from
  // the latest committed yaml every time they click 编辑.
  useEffect(() => {
    if (open) setRows(initial.length ? initial : [EMPTY_PROMPT])
  }, [open, initial])

  function patchRow(idx: number, patch: Partial<SamplingPromptValue>) {
    setRows((prev) => prev.map((row, i) => (i === idx ? { ...row, ...patch } : row)))
  }
  function deleteRow(idx: number) {
    setRows((prev) => prev.filter((_, i) => i !== idx))
  }
  function addRow() {
    setRows((prev) => [...prev, { ...EMPTY_PROMPT }])
  }

  function commit() {
    // Drop fully-empty rows so saving doesn't carry placeholder
    // rubbish into the recipe — but keep partial rows so a half-typed
    // negative / dim doesn't quietly disappear.
    const cleaned = rows.filter((r) => r.prompt.trim().length > 0)
    onSave(cleaned)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>编辑采样提示词</DialogTitle>
          <DialogDescription>
            训练过程中按计划生成预览图。每条提示词可设独立的负向提示、CFG、采样步数、种子、分辨率；留空时使用全局值。
            种子填 <code>-1</code> 表示每次运行随机抽取，掷骰按钮可立即生成新值。
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          {rows.length === 0 ? (
            <div className="text-center text-sm text-muted-foreground py-8">
              还没有提示词。
            </div>
          ) : (
            rows.map((row, idx) => (
              <PromptRow
                key={idx}
                index={idx}
                value={row}
                defaultResolution={defaultResolution}
                onChange={(p) => patchRow(idx, p)}
                onDelete={() => deleteRow(idx)}
              />
            ))
          )}
        </div>

        <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2 pt-3 border-t border-border/40">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={addRow}
            className="gap-1"
          >
            <Plus className="size-3.5" />
            添加一条
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onOpenChange(false)}
            >
              取消
            </Button>
            <Button size="sm" onClick={commit}>
              保存（{rows.filter((r) => r.prompt.trim()).length} 条）
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
})

interface PromptRowProps {
  index: number
  value: SamplingPromptValue
  defaultResolution?: [number, number] | undefined
  onChange: (patch: Partial<SamplingPromptValue>) => void
  onDelete: () => void
}

const PromptRow = memo(function PromptRow({
  index,
  value,
  defaultResolution,
  onChange,
  onDelete,
}: PromptRowProps) {
  const isRandomSeed =
    value.seed === -1 || value.seed === null || value.seed === undefined
  const [defW, defH] = defaultResolution ?? [1024, 1024]

  return (
    <div className="rounded-[5px] border border-border/50 bg-muted/15 p-3 space-y-2.5">
      <div className="flex items-start gap-2">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80 mt-1 pt-0.5 w-9 shrink-0">
          #{String(index + 1).padStart(2, "0")}
        </div>
        <textarea
          value={value.prompt}
          onChange={(e) => onChange({ prompt: e.target.value })}
          placeholder="正向提示词，例如：1girl, red hair, smiling, masterpiece"
          rows={2}
          className="flex-1 rounded-[4px] border border-input bg-background px-2.5 py-1.5 font-mono text-xs resize-y min-h-[44px]"
        />
        <button
          type="button"
          onClick={onDelete}
          className="mt-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[4px] border border-border/50 bg-background hover:bg-destructive/10 hover:text-destructive hover:border-destructive/40 text-muted-foreground"
          title="删除此条"
        >
          <Trash2 className="size-3.5" />
        </button>
      </div>

      <textarea
        value={value.negative ?? ""}
        onChange={(e) => onChange({ negative: e.target.value || null })}
        placeholder="负向提示词（可选）"
        rows={1}
        className="w-full rounded-[4px] border border-input bg-background px-2.5 py-1.5 font-mono text-xs resize-y min-h-[32px]"
      />

      <div className="grid grid-cols-[auto_1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] items-center gap-x-1.5 gap-y-1.5 text-[11px]">
        <Label className="text-muted-foreground">CFG</Label>
        <Input
          type="number"
          step="any"
          value={value.cfg ?? ""}
          placeholder="默认"
          onChange={(e) =>
            onChange({
              cfg: e.target.value === "" ? null : parseFloat(e.target.value),
            })
          }
          className="h-7 font-mono text-[11px]"
        />
        <Label className="text-muted-foreground">Steps</Label>
        <Input
          type="number"
          min={1}
          value={value.steps ?? ""}
          placeholder="默认"
          onChange={(e) =>
            onChange({
              steps: e.target.value === "" ? null : parseInt(e.target.value, 10),
            })
          }
          className="h-7 font-mono text-[11px]"
        />
        <Label className="text-muted-foreground">W</Label>
        <Input
          type="number"
          min={64}
          value={value.width ?? ""}
          placeholder={String(defW)}
          onChange={(e) =>
            onChange({
              width: e.target.value === "" ? null : parseInt(e.target.value, 10),
            })
          }
          className="h-7 font-mono text-[11px]"
        />
        <Label className="text-muted-foreground">H</Label>
        <Input
          type="number"
          min={64}
          value={value.height ?? ""}
          placeholder={String(defH)}
          onChange={(e) =>
            onChange({
              height: e.target.value === "" ? null : parseInt(e.target.value, 10),
            })
          }
          className="h-7 font-mono text-[11px]"
        />
        <Label className="text-muted-foreground">Seed</Label>
        <div className="flex items-center gap-1">
          <Input
            type="number"
            min={-1}
            max={SEED_MAX}
            value={isRandomSeed ? "" : String(value.seed)}
            placeholder="随机 (-1)"
            onChange={(e) => {
              const raw = e.target.value.trim()
              if (raw === "" || raw === "-1") {
                onChange({ seed: -1 })
                return
              }
              const n = parseInt(raw, 10)
              if (!Number.isNaN(n)) {
                onChange({ seed: Math.max(-1, Math.min(SEED_MAX, n)) })
              }
            }}
            className="h-7 font-mono text-[11px] tabular-nums"
          />
          <button
            type="button"
            onClick={() =>
              onChange({ seed: Math.floor(Math.random() * SEED_MAX) })
            }
            title="掷骰子（生成新种子并固定）"
            className="inline-flex h-7 w-7 items-center justify-center rounded-[4px] border border-border/50 bg-background hover:bg-muted/40 text-sm"
          >
            <Shuffle className="size-3" />
          </button>
        </div>
      </div>
    </div>
  )
})
