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
  // config with a single-value list doesn't crash the dialog.
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
            description="对应 kohya 的 --sample_every_n_steps；与 everyNEpochs 不冲突。留空则仅按回合采样。"
            errors={errorMap.get("sampling.everyNSteps")}
          >
            <IntInput
              min={1}
              value={v.everyNSteps ?? null}
              onChange={(n) => set(["sampling", "everyNSteps"], n)}
              placeholder="默认"
            />
          </Row>
          <Row
            label="训练前先采样"
            description="对应 kohya 的 --sample_at_first：第 0 步即生成一组基线样图。"
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
                ? `已定义 ${prompts.length} 条提示词，全部存于 yaml；启动时自动写出 prompts.txt。`
                : "通过对话框逐条编辑提示词；保存至 yaml 后，启动训练时自动生成 prompts.txt。"
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
              {prompts.length > 0 ? `编辑 · ${prompts.length}` : "编辑提示词"}
            </Button>
          </Row>
          {/* Legacy field — kept visible in case the user wants to point
              at an existing prompts.txt instead of authoring inline. */}
          <Row
            label="提示词文件"
            description="可选 · 指向外部 prompts.txt；留空则使用上方对话框中的列表。"
            errors={errorMap.get("sampling.promptsFile")}
          >
            <PathInput
              value={v.promptsFile ?? ""}
              onChange={(s) => set(["sampling", "promptsFile"], s || null)}
              placeholder="可选 · ./prompts.txt"
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
            description="-1 = 每次训练随机抽取（与 ComfyUI 同义）；填具体数字即固定种子；🎲 立刻生成新值并固定。"
          >
            <SeedInput
              value={v.seed ?? -1}
              onChange={(n) => set(["sampling", "seed"], n)}
            />
          </Row>

          <Row
            label="触发词"
            description="提示词中的 ${TRIGGER} 占位符将在训练启动时替换为此值。留空时，自动从数据集 .txt 描述文件推断（取第一个 token 的众数，过滤 1girl / masterpiece 等通用标签）；推断失败则连同相邻逗号一并删除占位符。"
          >
            <TextInput
              value={v.triggerWord ?? ""}
              onChange={(s) => set(["sampling", "triggerWord"], s || null)}
              placeholder="例如 thornsdance · 留空自动推断"
              className="w-72"
            />
          </Row>

          <Row
            label="预览输出"
            description="训练过程中附加生成的预览图产物，按需勾选。"
          >
            <div className="flex flex-col gap-1.5 text-[12px]">
              <OutputToggle
                label="多 Prompt 网格图"
                description="每个 ckpt 将所有提示词的产出横向拼接成一张总览图。"
                checked={outputs.gridStitching ?? true}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "gridStitching"], b)
                }
              />
              <OutputToggle
                label="基模对比"
                description="同一提示词额外渲染一份不挂 LoRA 的基模产出，便于直观对比 LoRA 学习效果。GPU 耗时翻倍。"
                checked={outputs.baseCompare ?? false}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "baseCompare"], b)
                }
              />
              <OutputToggle
                label="跨 Ckpt 动画"
                description="将同一提示词在各 ckpt 上的产出合成 GIF，可视化训练过程演变。"
                checked={outputs.crossCkptAnimation ?? false}
                onCheckedChange={(b) =>
                  set(["sampling", "outputs", "crossCkptAnimation"], b)
                }
              />
              <OutputToggle
                label="PNG 元数据"
                description="将 prompt / seed / step / cfg 写入 PNG parameters 区，兼容 A1111 / ComfyUI。"
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
        triggerWord={v.triggerWord ?? null}
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
