import { memo, useState } from "react"
import { Pencil } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ErrorMap, ConfigFormValue, Setter, SamplingPromptValue } from "../types"
import {
  IntInput,
  PathInput,
  ResolutionInput,
  Row,
  SeedInput,
  TextInput,
  ToggleSwitch,
} from "../widgets"
import { PromptsDialog } from "./prompts-dialog"

export const SamplingFields = memo(function SamplingFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["sampling"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const enabled = v.enabled ?? true
  const prompts = v.prompts ?? []
  const outputs = v.outputs ?? {}
  const [promptsOpen, setPromptsOpen] = useState(false)

  // Resolution lives as an arbitrary-length list in yaml but the form
  // ResolutionInput takes a [w, h] tuple. Coerce defensively so an old
  // recipe with a single-value list doesn't crash the dialog.
  const resTuple: [number, number] | undefined =
    Array.isArray(v.resolution) && v.resolution.length >= 2
      ? [v.resolution[0]!, v.resolution[1]!]
      : undefined

  return (
    <>
      <Row label="启用采样" description="训练过程中生成预览图。">
        <ToggleSwitch
          checked={enabled}
          onCheckedChange={(b) => set(["sampling", "enabled"], b)}
        />
      </Row>
      {enabled && (
        <>
          <Row label="每 N 回合一次">
            <IntInput
              min={1}
              value={v.everyNEpochs ?? 1}
              onChange={(n) => set(["sampling", "everyNEpochs"], n ?? 1)}
            />
          </Row>
          <Row
            label="每 N 步一次"
            description="kohya `--sample_every_n_steps`；与 everyNEpochs 互不冲突。留空仅按回合采样。"
            errors={errorMap.get("sampling.everyNSteps")}
          >
            <IntInput
              min={1}
              value={v.everyNSteps ?? null}
              onChange={(n) => set(["sampling", "everyNSteps"], n)}
              placeholder="（默认）"
            />
          </Row>
          <Row
            label="训练前先采样"
            description="kohya `--sample_at_first`：第 0 步生成一组基线样图。"
          >
            <ToggleSwitch
              checked={v.atFirst ?? false}
              onCheckedChange={(b) => set(["sampling", "atFirst"], b)}
            />
          </Row>
          <Row
            label="提示词"
            description={
              prompts.length > 0
                ? `已定义 ${prompts.length} 条提示词，全部保存在 yaml 中，启动时自动写出 prompts.txt。`
                : "通过对话框逐条编辑提示词；保存到 yaml 后启动训练时会自动生成 prompts.txt。"
            }
          >
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setPromptsOpen(true)}
              className="gap-1.5"
            >
              <Pencil className="size-3.5" />
              {prompts.length > 0 ? `编辑（${prompts.length}）` : "编辑提示词"}
            </Button>
          </Row>
          {/* Legacy field — kept visible in case the user wants to point
              at an existing prompts.txt instead of authoring inline. */}
          <Row
            label="提示词文件（可选）"
            description="指向外部 prompts.txt；留空则使用上方对话框中的列表。"
            errors={errorMap.get("sampling.promptsFile")}
          >
            <PathInput
              value={v.promptsFile ?? ""}
              onChange={(s) => set(["sampling", "promptsFile"], s || null)}
              placeholder="（可选）./prompts.txt"
            />
          </Row>
          <Row label="分辨率">
            <ResolutionInput
              value={v.resolution}
              onChange={(r) => set(["sampling", "resolution"], r)}
            />
          </Row>
          <Row
            label="随机种子"
            description="-1 = 每次训练随机抽取（与 ComfyUI 相同语义）；填具体数字即固定种子；🎲 立刻生成新值并固定。"
          >
            <SeedInput
              value={v.seed ?? -1}
              onChange={(n) => set(["sampling", "seed"], n)}
            />
          </Row>

          <Row
            label="触发词"
            description="提示词里写 ${TRIGGER} 占位符，启动训练时自动替换为此值。留空 → 自动从 dataset 的 .txt caption 推断（取第一个 token 的众数，过滤 1girl/masterpiece 等通用标签）；都拿不到 → 占位符与邻接的逗号一起去掉。"
          >
            <TextInput
              value={v.triggerWord ?? ""}
              onChange={(s) => set(["sampling", "triggerWord"], s || null)}
              placeholder="例如 thornsdance（留空自动推断）"
              className="w-72"
            />
          </Row>

          <Row
            label="预览输出"
            description="训练时附加生成的预览图产物，按需勾选。"
          >
            <div className="flex flex-col gap-1.5 text-[12px]">
              <OutputToggle
                label="多 prompt 网格图"
                description="每个 ckpt 把所有提示词的产出横向拼接成一张总览图。"
                checked={outputs.gridStitching ?? true}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "gridStitching"], b)
                }
              />
              <OutputToggle
                label="基模对比"
                description="同提示词额外渲染一份不挂载 LoRA 的基模产出，便于一眼看出 LoRA 的影响。会双倍 GPU 时间。"
                checked={outputs.baseCompare ?? false}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "baseCompare"], b)
                }
              />
              <OutputToggle
                label="跨 ckpt 动画"
                description="把同一条提示词在所有 ckpt 上的产出累成 gif，可滑动查看训练过程的演变。"
                checked={outputs.crossCkptAnimation ?? false}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "crossCkptAnimation"], b)
                }
              />
              <OutputToggle
                label="PNG 元数据"
                description="把 prompt / 种子 / step / cfg 写进 PNG 的 parameters 区，与 A1111 / ComfyUI 兼容。"
                checked={outputs.pngMetadata ?? true}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "pngMetadata"], b)
                }
              />
            </div>
          </Row>
        </>
      )}

      <PromptsDialog
        open={promptsOpen}
        onOpenChange={setPromptsOpen}
        initial={prompts}
        defaultResolution={resTuple}
        onSave={(next: SamplingPromptValue[]) =>
          set(["sampling", "prompts"], next)
        }
      />
    </>
  )
})

const OutputToggle = memo(function OutputToggle({
  label,
  description,
  checked,
  onCheckedChange,
}: {
  label: string
  description: string
  checked: boolean
  onCheckedChange: (b: boolean) => void
}) {
  return (
    <label className="flex items-start gap-2 cursor-pointer select-none">
      <ToggleSwitch
        checked={checked}
        onCheckedChange={onCheckedChange}
      />
      <span className="flex flex-col leading-tight">
        <span className="text-foreground">{label}</span>
        <span className="text-[11px] text-muted-foreground">{description}</span>
      </span>
    </label>
  )
})
