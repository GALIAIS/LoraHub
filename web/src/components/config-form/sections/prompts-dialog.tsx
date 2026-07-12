/**
 * 采样提示词编辑对话框。
 *
 * 用户在主表单点击「编辑提示词」按钮即弹出，提供一行一条提示词的
 * 编辑界面：每行支持独立的 negative / cfg / steps / seed / w×h /
 * sampler / flow_shift。保存时整体写回 ConfigFormValue.sampling.prompts；
 * 后端在任务启动时把这份列表物化成 kohya 风格的 prompts.txt（见
 * lifecycle._materialise_prompts_file），所以前端不依赖外部 .txt
 * 文件，所有内容直接保存在 yaml 中。
 *
 * Anima flow-matching 的 sampler 仅有 euler / er_sde / lcm 三种；
 * scheduler 不暴露——上游用 flow_shift（默认 5.0）单参数控制 schedule。
 *
 * 种子语义与 SeedInput 保持一致：-1 表示「运行时随机」。
 */
import { memo, useEffect, useMemo, useState } from "react"
import { ChevronDown, ChevronRight, Plus, Shuffle, Trash2 } from "lucide-react"
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import type { SamplingPromptValue } from "../types"
import { FloatInput } from "../widgets"

const SEED_MAX = 1_125_899_906_842_624

const EMPTY_PROMPT: SamplingPromptValue = {
  prompt: "",
  negative: null,
  cfg: null,
  steps: null,
  seed: -1,
  width: null,
  height: null,
  sampler: null,
  flowShift: null,
}

// 上游 anima_lora/inference.py:209 — flow-matching 仅这三种.
const SAMPLER_OPTIONS = [
  { value: "euler", label: "euler · 确定性 ODE（默认）" },
  { value: "er_sde", label: "er_sde · 扩展逆时 SDE" },
  { value: "lcm", label: "lcm · x0 重噪（蒸馏少步模型）" },
] as const

// 占位符与 SamplingConfig.trigger_word 的语义一致。
const TRIGGER_PLACEHOLDER = "${TRIGGER}"

function foldPromptText(value: string): string {
  return value.split(/\s+/u).filter(Boolean).join(" ")
}

export interface PromptsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initial: SamplingPromptValue[]
  defaultResolution?: [number, number] | undefined
  /** Trigger word from sampling.triggerWord; null = 留空（启动时自动推断/剥离）。*/
  triggerWord?: string | null
  showAdvanced?: boolean
  onSave: (next: SamplingPromptValue[]) => void
}

export const PromptsDialog = memo(function PromptsDialog({
  open,
  onOpenChange,
  initial,
  defaultResolution,
  triggerWord,
  showAdvanced = true,
  onSave,
}: PromptsDialogProps) {
  const [rows, setRows] = useState<SamplingPromptValue[]>(initial)

  useEffect(() => {
    if (open) setRows(initial.length ? initial : [{ ...EMPTY_PROMPT }])
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
    const cleaned = rows
      .map((row) => {
        const prompt = foldPromptText(row.prompt)
        const negative = foldPromptText(row.negative ?? "")
        return { ...row, prompt, negative: negative || null }
      })
      .filter((row) => row.prompt.length > 0)
    onSave(cleaned)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[88vh] overflow-hidden flex flex-col">
        <DialogHeader className="space-y-1.5">
          <DialogTitle>编辑采样提示词</DialogTitle>
          <DialogDescription className="text-xs leading-relaxed">
            训练过程中按计划生成预览图。每条 prompt 可独立设置基础参数与高级参数（sampler / flow_shift），留空时使用全局默认。
            <br />
            <code className="text-[10.5px]">{TRIGGER_PLACEHOLDER}</code> 占位符在启动时替换为
            {triggerWord ? (
              <>
                {" "}<span className="font-mono font-medium text-primary">{triggerWord}</span>。
              </>
            ) : (
              <> 触发词字段（当前未设，将自动从数据集推断或剥离占位符）。</>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-2.5 pr-2 -mr-2 py-1">
          {rows.length === 0 ? (
            <div className="text-center text-sm text-muted-foreground py-12">
              还没有提示词，点击下方「添加一条」开始。
            </div>
          ) : (
            rows.map((row, idx) => (
              <PromptRow
                key={idx}
                index={idx}
                value={row}
                defaultResolution={defaultResolution}
                triggerWord={triggerWord ?? null}
                showAdvanced={showAdvanced}
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
  triggerWord?: string | null
  showAdvanced: boolean
  onChange: (patch: Partial<SamplingPromptValue>) => void
  onDelete: () => void
}

const PromptRow = memo(function PromptRow({
  index,
  value,
  defaultResolution,
  triggerWord,
  showAdvanced,
  onChange,
  onDelete,
}: PromptRowProps) {
  const isRandomSeed =
    value.seed === -1 || value.seed === null || value.seed === undefined
  const [defW, defH] = defaultResolution ?? [1024, 1024]

  // 「展开后实际 prompt」预览：占位符 → 触发词字段 / 自动剥离提示。
  const resolvedPreview = useMemo(() => {
    const body = value.prompt
    if (!body.includes(TRIGGER_PLACEHOLDER)) return null
    if (triggerWord && triggerWord.trim()) {
      return body.replaceAll(TRIGGER_PLACEHOLDER, triggerWord.trim())
    }
    // 与后端 _resolve_trigger_word + _TRIGGER_PLACEHOLDER_WITH_GLUE_RE 行为一致：
    // 占位符及相邻逗号一并剥离。
    return body.replace(/(,\s*)?\$\{TRIGGER\}(\s*,)?/g, "").replace(/^,\s*|,\s*$/g, "")
  }, [value.prompt, triggerWord])

  const [advancedOpen, setAdvancedOpen] = useState(
    value.sampler != null || value.flowShift != null,
  )

  return (
    <div className="rounded-[6px] border border-border/50 bg-muted/15 p-3 space-y-2.5 hover:border-border/70 transition-colors">
      {/* Header: 序号 + 删除 */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80">
          PROMPT #{String(index + 1).padStart(2, "0")}
        </span>
        <button
          type="button"
          onClick={onDelete}
          className="inline-flex h-6 items-center gap-1 rounded-[4px] border border-border/40 bg-background px-2 text-[10.5px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive hover:border-destructive/40"
          title="删除此条"
        >
          <Trash2 className="size-3" />
          删除
        </button>
      </div>

      {/* 正向 prompt */}
      <textarea
        value={value.prompt}
        onChange={(e) => onChange({ prompt: e.target.value })}
        placeholder={`正向提示词，例如：${TRIGGER_PLACEHOLDER}, 1girl, red hair, smiling, masterpiece`}
        rows={2}
        className="w-full rounded-[4px] border border-input bg-background px-2.5 py-1.5 font-mono text-xs resize-y min-h-[48px]"
      />

      {/* 触发词替换预览 */}
      {resolvedPreview !== null && (
        <div className="rounded-[4px] border border-dashed border-primary/30 bg-primary/5 px-2.5 py-1.5">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[9.5px] uppercase tracking-wider text-primary/80 shrink-0">
              展开后
            </span>
            <span className="font-mono text-[11px] text-foreground/85 break-words">
              {resolvedPreview || <em className="text-muted-foreground">（占位符已剥离）</em>}
            </span>
          </div>
        </div>
      )}

      {/* 负向 prompt */}
      <textarea
        value={value.negative ?? ""}
        onChange={(e) => onChange({ negative: e.target.value || null })}
        placeholder="负向提示词（可选）"
        rows={1}
        className="w-full rounded-[4px] border border-input bg-background px-2.5 py-1.5 font-mono text-xs resize-y min-h-[32px]"
      />

      {/* 基础参数行：CFG / Steps / W / H / Seed */}
      <div className="grid grid-cols-5 gap-2">
        <Field label="CFG">
          <FloatInput
            value={value.cfg ?? null}
            placeholder="默认"
            onChange={(cfg) => onChange({ cfg })}
            className="h-7 w-full text-[11px]"
          />
        </Field>
        <Field label="Steps">
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
        </Field>
        <Field label="Width">
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
        </Field>
        <Field label="Height">
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
        </Field>
        <Field label="Seed">
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
              className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[4px] border border-border/50 bg-background hover:bg-muted/40 text-sm"
            >
              <Shuffle className="size-3" />
            </button>
          </div>
        </Field>
      </div>

      {/* 高级参数：sampler + flow_shift（默认折叠） */}
      {showAdvanced && (
        <>
          <button
            type="button"
            onClick={() => setAdvancedOpen((v) => !v)}
            className={cn(
              "flex items-center gap-1 text-[10.5px] font-medium uppercase tracking-wider",
              "text-muted-foreground hover:text-foreground transition-colors",
            )}
          >
            {advancedOpen ? (
              <ChevronDown className="size-3" />
            ) : (
              <ChevronRight className="size-3" />
            )}
            高级 · sampler / flow_shift
            {(value.sampler != null || value.flowShift != null) && !advancedOpen && (
              <span className="ml-1 rounded-full bg-primary/15 px-1.5 py-px text-[9px] text-primary">
                已设置
              </span>
            )}
          </button>

          {advancedOpen && (
            <div className="grid grid-cols-2 gap-2 pt-0.5">
              <Field label="Sampler">
                <Select
                  value={value.sampler ?? "_default"}
                  onValueChange={(s) =>
                    onChange({
                      sampler:
                        s === "_default"
                          ? null
                          : (s as "euler" | "er_sde" | "lcm"),
                    })
                  }
                >
                  <SelectTrigger className="h-7 text-[11px] font-mono">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="_default" className="text-[11px]">
                      默认（euler）
                    </SelectItem>
                    {SAMPLER_OPTIONS.map((o) => (
                      <SelectItem key={o.value} value={o.value} className="text-[11px]">
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Flow Shift">
                <FloatInput
                  step={0.1}
                  min={0.01}
                  value={value.flowShift ?? null}
                  placeholder="默认 5.0"
                  onChange={(flowShift) => onChange({ flowShift })}
                  className="h-7 w-full text-[11px]"
                />
              </Field>
            </div>
          )}
        </>
      )}
    </div>
  )
})

const Field = memo(function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label className="text-[10px] uppercase tracking-wider text-muted-foreground/85">
        {label}
      </Label>
      {children}
    </div>
  )
})
