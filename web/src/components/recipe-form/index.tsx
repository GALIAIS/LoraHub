/**
 * Visual recipe editor — covers every field that influences training.
 *
 * Built directly against RecipeConfig (lorahub/core/config/schema.py) so each
 * widget knows its semantics; the form is collapsible per section, validates
 * locally, and surfaces server validation errors next to the offending field.
 */
import { useCallback, useMemo } from "react"
import {
  Activity,
  Cpu,
  FileImage,
  Folder,
  Gauge,
  History,
  Image,
  Layers,
  PaintBucket,
  Settings2,
  SlidersHorizontal,
  Wand2,
  Workflow,
} from "lucide-react"
import type { ValidationFieldError } from "@/lib/api"
import { BaseModelFields } from "./sections/base-model"
import { DatasetFields } from "./sections/dataset"
import { NetworkFields } from "./sections/network"
import { OptimizerFields } from "./sections/optimizer"
import { LossFields } from "./sections/loss"
import { ScheduleFields } from "./sections/schedule"
import { PrecisionFields } from "./sections/precision"
import { SamplingFields } from "./sections/sampling"
import { OutputFields } from "./sections/output"
import { BackendFields } from "./sections/backend"
import { BackendDiffusionPipeFields } from "./sections/backend-diffusion-pipe"
import { ValidationFields } from "./sections/validation"
import { ResumeFields } from "./sections/resume"
import { buildErrorMap, setIn } from "./types"
import type { RecipeFormValue } from "./types"
import { ReadOnlyProvider, Section } from "./widgets"

export type { RecipeFormValue } from "./types"

interface RecipeFormProps {
  value: RecipeFormValue
  onChange: (next: RecipeFormValue) => void
  errors?: ValidationFieldError[]
  /**
   * When true the form is rendered for browsing only — every input is
   * disabled (via a wrapping <fieldset disabled> plus a context flag for the
   * widgets that don't naturally inherit it). Used by RecipePreview so the
   * recipe is rendered as a structured overview instead of raw YAML.
   */
  readOnly?: boolean
}

export function RecipeForm({ value, onChange, errors, readOnly = false }: RecipeFormProps) {
  const errorMap = useMemo(() => buildErrorMap(errors), [errors])

  // Stable updater factory to keep child callbacks identity-stable when
  // value/onChange themselves are stable.
  const set = useCallback(
    (path: ReadonlyArray<string | number>, next: unknown) => {
      onChange(setIn(value, path, next))
    },
    [value, onChange],
  )

  const body = (
    <div className="space-y-3">
      <Section
        icon={<Cpu className="size-3.5" />}
        title="基础模型"
        subtitle="选择架构与待微调的 .safetensors 检查点"
        defaultOpen
      >
        <BaseModelFields value={value.base_model} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<FileImage className="size-3.5" />}
        title="数据集"
        subtitle="训练图片的位置与加载方式"
        defaultOpen
      >
        <DatasetFields value={value.dataset} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Layers className="size-3.5" />}
        title="网络"
        subtitle="LoRA 结构：rank、alpha、目标模块"
        defaultOpen
      >
        <NetworkFields value={value.network} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<SlidersHorizontal className="size-3.5" />}
        title="优化器与学习率"
        subtitle="权重的更新策略"
      >
        <OptimizerFields value={value.optimizer} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Activity className="size-3.5" />}
        title="损失整形"
        subtitle="Min-SNR、噪声偏移、loss_type 等"
      >
        <LossFields value={value.loss} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Settings2 className="size-3.5" />}
        title="训练计划"
        subtitle="回合数、批大小、梯度累积"
        defaultOpen
      >
        <ScheduleFields value={value.schedule} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<PaintBucket className="size-3.5" />}
        title="精度与显存"
        subtitle="混合精度、梯度检查点、潜变量缓存"
      >
        <PrecisionFields value={value} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Gauge className="size-3.5" />}
        title="验证"
        subtitle="留出比例与验证频率"
      >
        <ValidationFields value={value} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Image className="size-3.5" />}
        title="采样预览"
        subtitle="训练过程中定期生成预览图"
      >
        <SamplingFields value={value.sampling} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Folder className="size-3.5" />}
        title="输出"
        subtitle="文件名、保存频率、保存精度"
        defaultOpen
      >
        <OutputFields value={value.output} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<History className="size-3.5" />}
        title="断点续训"
        subtitle="optimizer / scheduler state 落盘策略"
      >
        <ResumeFields value={value.resume} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Wand2 className="size-3.5" />}
        title="后端覆盖"
        subtitle="按配置覆盖 kohya 检出目录与 Python 解释器"
      >
        <BackendFields value={value.backend} set={set} errorMap={errorMap} />
      </Section>

      {value.backend?.type === "diffusion-pipe" && (
        <Section
          icon={<Workflow className="size-3.5" />}
          title="diffusion-pipe 选项"
          subtitle="仅在 diffusion-pipe 后端下生效"
        >
          <BackendDiffusionPipeFields
            value={value.backend?.diffusion_pipe}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}
    </div>
  )

  if (!readOnly) return <ReadOnlyProvider value={false}>{body}</ReadOnlyProvider>
  return (
    <ReadOnlyProvider value={true}>
      <fieldset
        disabled
        className="border-0 p-0 m-0 min-w-0 disabled:opacity-100"
      >
        {body}
      </fieldset>
    </ReadOnlyProvider>
  )
}
