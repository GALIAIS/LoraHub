import { memo, useState } from "react"
import { Pencil } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ErrorMap, ConfigFormValue, Setter, SamplingPromptValue } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  PathInput,
  ResolutionInput,
  Row,
  SeedInput,
  TextInput,
  ToggleSwitch,
} from "../widgets"
import { PromptsDialog } from "./prompts-dialog"

const SAMPLE_SAMPLER_OPTIONS = [
  { value: "ddim", label: "DDIM" },
  { value: "pndm", label: "PNDM" },
  { value: "lms", label: "LMS" },
  { value: "euler", label: "Euler" },
  { value: "euler_a", label: "Euler a" },
  { value: "heun", label: "Heun" },
  { value: "dpm_2", label: "DPM 2" },
  { value: "dpm_2_a", label: "DPM 2 a" },
  { value: "dpmsolver", label: "DPMSolver" },
  { value: "dpmsolver++", label: "DPMSolver++" },
  { value: "dpmsingle", label: "DPMSingle" },
  { value: "k_lms", label: "K-LMS" },
  { value: "k_euler", label: "K-Euler" },
  { value: "k_euler_a", label: "K-Euler a" },
  { value: "k_dpm_2", label: "K-DPM 2" },
  { value: "k_dpm_2_a", label: "K-DPM 2 a" },
] as const

export const SamplingFields = memo(function SamplingFields({
  value = {},
  set,
  errorMap,
  backendType,
}: {
  value: ConfigFormValue["sampling"]
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const v = value ?? {}
  const enabled = v.enabled ?? true
  const prompts = v.prompts ?? []
  const outputs = v.outputs ?? {}
  const isAiToolkit = backendType === "ai_toolkit"
  const isDiffusionPipe = backendType === "diffusion-pipe"
  const isAnima = backendType === "anima_lora"
  const defaultSampler = backendType === "kohya" ? "euler_a" : "ddim"
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
      <Row label="检查点频谱分析" description="保存检查点后分析适配器权重频谱。">
        <ToggleSwitch
          checked={v.spectrumAnalysis ?? true}
          onCheckedChange={(b) => set(["sampling", "spectrumAnalysis"], b)}
        />
      </Row>
      {enabled && (
        <>
          {isDiffusionPipe && (
            <Row label="实时检查点预览" description="检测新检查点并在独立推理进程中生成预览图。">
              <ToggleSwitch
                checked={v.enableLiveInference ?? false}
                onCheckedChange={(b) => set(["sampling", "enableLiveInference"], b)}
              />
            </Row>
          )}
          {!isDiffusionPipe && (
            <>
              <Row
                label="每 N 回合一次"
                description={isAiToolkit ? "每完成 N 个训练数据回合生成一次预览，可与按步采样同时使用。" : "每完成 N 个训练回合生成一次预览。"}
                errors={errorMap.get("sampling.everyNEpochs")}
              >
                <IntInput min={1} value={v.everyNEpochs ?? 1} onChange={(n) => set(["sampling", "everyNEpochs"], n ?? 1)} />
              </Row>
              <Row
                label="每 N 步一次"
                description={isAiToolkit ? "可选。与按回合采样同时使用；留空则仅按回合采样。" : "与回合采样并行生效；留空则仅按回合采样。"}
                errors={errorMap.get("sampling.everyNSteps")}
              >
                <IntInput min={1} value={v.everyNSteps ?? null} onChange={(n) => set(["sampling", "everyNSteps"], n)} placeholder="默认" />
              </Row>
            </>
          )}
          {!isAiToolkit && !isDiffusionPipe && (
            <>
              <Row
                label="训练前先采样"
                description="第 0 步生成一组基线样图。"
              >
                <ToggleSwitch
                  checked={v.atFirst ?? false}
                  onCheckedChange={(b) => set(["sampling", "atFirst"], b)}
                />
              </Row>
              <Row
                label="预览采样器"
                description="留空使用后端默认采样器。"
                errors={errorMap.get("sampling.sampleSampler")}
              >
                <EnumSelect
                  value={v.sampleSampler ?? defaultSampler}
                  onChange={(s) => set(["sampling", "sampleSampler"], s)}
                  options={SAMPLE_SAMPLER_OPTIONS}
                />
              </Row>
            </>
          )}
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
          {!isAiToolkit && (
            <Row
              label="提示词文件"
              description="可选。留空则使用上方提示词列表。"
              errors={errorMap.get("sampling.promptsFile")}
            >
              <PathInput
                value={v.promptsFile ?? ""}
                onChange={(s) => set(["sampling", "promptsFile"], s || null)}
                placeholder="可选 · ./prompts.txt"
              />
            </Row>
          )}
          <Row label="分辨率">
            <ResolutionInput
              value={v.resolution}
              onChange={(r) => set(["sampling", "resolution"], r)}
            />
          </Row>
          {(isAiToolkit || isDiffusionPipe || isAnima) && (
            <>
              <Row
                label="采样步数"
                description="预览推理步数。"
                errors={errorMap.get("sampling.inferenceSteps")}
              >
                <IntInput
                  min={1}
                  value={v.inferenceSteps ?? 24}
                  onChange={(n) => set(["sampling", "inferenceSteps"], n ?? 24)}
                />
              </Row>
              <Row
                label="CFG"
                description="预览推理引导强度。"
                errors={errorMap.get("sampling.inferenceCfg")}
              >
                <FloatInput
                  min={0.1}
                  step={0.1}
                  value={v.inferenceCfg ?? 5}
                  onChange={(n) => set(["sampling", "inferenceCfg"], n ?? 5)}
                />
              </Row>
            </>
          )}
          <Row
            label="随机种子"
            description="-1 = 每次训练随机抽取（与 ComfyUI 同义）；填具体数字即固定种子；骰子按钮会立刻生成新值并固定。"
          >
            <SeedInput
              value={v.seed ?? -1}
              onChange={(n) => set(["sampling", "seed"], n)}
            />
          </Row>

          {!isAiToolkit && (
            <Row
              label="触发词"
              description="训练启动时替换提示词中的 ${TRIGGER}。"
            >
              <TextInput
                value={v.triggerWord ?? ""}
                onChange={(s) => set(["sampling", "triggerWord"], s || null)}
                placeholder="可选"
                className="w-72"
              />
            </Row>
          )}

          {!isAiToolkit && (
            <Row
              label="预览输出"
              description="训练过程中附加生成的预览图产物。"
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
                  description="同一提示词额外渲染一份不挂 LoRA 的基模产出，用于对比。"
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
                  description="将 prompt / seed / step / cfg 写入 PNG parameters 区。"
                  checked={outputs.pngMetadata ?? true}
                  onCheckedChange={(b) =>
                    set(["sampling", "outputs", "pngMetadata"], b)
                  }
                />
              </div>
            </Row>
          )}
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
