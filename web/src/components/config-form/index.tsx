/**
 * Visual config editor — covers every field that influences training.
 *
 * Built directly against TrainingConfig (lorahub/core/config/schema.py) so each
 * widget knows its semantics; the form is collapsible per section, validates
 * locally, and surfaces server validation errors next to the offending field.
 *
 * Backend-specific sections gate on cfg.backend.type so users only see the
 * knobs the selected backend's compiler actually consumes — kohya reads
 * almost the full schema, diffusion-pipe reads most of it, and anima_lora
 * reads only its dedicated AnimaLoraOptions sub-tree (plus dataset / output
 * / sampling). Hidden sections keep their values in form state so flipping
 * backend back and forth never silently drops user input.
 */
import { useCallback, useMemo } from "react"
import {
  Activity,
  Cpu,
  Database,
  FileImage,
  Flame,
  Folder,
  Gauge,
  History,
  Image,
  Layers,
  PaintBucket,
  Rocket,
  Settings2,
  Shuffle,
  SlidersHorizontal,
  Sparkles,
  LineChart,
  Wand2,
  Workflow,
  Zap,
} from "lucide-react"
import type { ValidationFieldError } from "@/lib/api"
import { ArchPathsFields } from "./sections/arch-paths"
import { AttentionFields } from "./sections/attention"
import { AdvancedLossFields } from "./sections/advanced-loss"
import { AugmentationFields } from "./sections/augmentation"
import { BaseModelFields } from "./sections/base-model"
import { DatasetFields } from "./sections/dataset"
import { DataLoaderFields } from "./sections/data-loader"
import { FlowMatchFields } from "./sections/flow-match"
import { NetworkFields } from "./sections/network"
import { OptimizerFields } from "./sections/optimizer"
import { LossFields } from "./sections/loss"
import { ScheduleFields } from "./sections/schedule"
import { PrecisionFields } from "./sections/precision"
import { OptimizationFields } from "./sections/optimization"
import { SamplingFields } from "./sections/sampling"
import { MonitoringFields } from "./sections/monitoring"
import { OutputFields } from "./sections/output"
import { BackendFields } from "./sections/backend"
import { BackendAnimaLoraFields } from "./sections/backend-anima-lora"
import { BackendDiffusionPipeFields } from "./sections/backend-diffusion-pipe"
import { ValidationFields } from "./sections/validation"
import { ResumeFields } from "./sections/resume"
import { FLOW_MATCH_ARCHES } from "./options"
import { buildErrorMap, setIn } from "./types"
import type { ConfigFormValue } from "./types"
import { ReadOnlyProvider, Section } from "./widgets"

export type { ConfigFormValue } from "./types"

// Which backends actually consume each section. Sections not listed here
// are universal (every backend reads them — basemodel / dataset / schedule /
// output / sampling / archPaths / backend overrides).
//
// Kept in sync with the compilers under lorahub/core/backends/<id>/. Update
// here whenever a compiler starts (or stops) reading a top-level field.
export type BackendKey = "kohya" | "diffusion-pipe" | "anima_lora"
const BACKENDS_ALL: readonly BackendKey[] = ["kohya", "diffusion-pipe", "anima_lora"]

export const SECTION_BACKENDS: Record<string, readonly BackendKey[]> = {
  network: ["kohya", "diffusion-pipe"], // anima reads backend.animaLora.networkDim/Alpha instead
  optimizer: ["kohya", "diffusion-pipe"], // anima reads backend.animaLora.optimizerType / learningRate
  loss: ["kohya", "diffusion-pipe"], // anima drives loss via flow-match weighting in animaLora.*
  advancedLoss: ["kohya"], // multires-noise / huber / pseudo-huber are kohya-only
  flowMatch: ["kohya", "diffusion-pipe"], // anima reads backend.animaLora.timestepSampling
  attention: ["kohya", "diffusion-pipe"], // anima reads backend.animaLora.attnMode
  precision: ["kohya", "diffusion-pipe"], // anima drives precision via animaLora.mixedPrecision
  optimization: ["kohya", "diffusion-pipe"], // anima exposes its own offload/compile knobs
  dataloader: ["kohya", "diffusion-pipe"], // anima ignores num_workers / vae_batch_size etc.
  augmentation: ["kohya", "diffusion-pipe"], // anima has no augmentation knobs at the top level
  validation: ["kohya", "diffusion-pipe"], // anima uses animaLora.useCmmd + validationSplitNum
  resume: ["kohya", "diffusion-pipe", "anima_lora"], // all three back ends read save_state*
  monitoring: ["kohya", "diffusion-pipe", "anima_lora"], // wandb is universal: see lorahub/api/wandb_env.py
}

/**
 * Return every top-level ConfigFormValue key that the given backend's compiler
 * does NOT consume. Caller can use this to strip orphan sections from a
 * payload right before save / launch, so the YAML doesn't carry stale
 * fields like ``network.rank=32`` next to ``backend.animaLora.networkDim=16``
 * (which is what actually feeds the trainer).
 */
export function unusedSectionsForBackend(backend: BackendKey | undefined): string[] {
  if (backend === undefined) return []
  return Object.entries(SECTION_BACKENDS)
    .filter(([, allowed]) => !allowed.includes(backend))
    .map(([section]) => section)
}

function showsForBackend(section: string, backend: BackendKey | undefined): boolean {
  const allowed = SECTION_BACKENDS[section] ?? BACKENDS_ALL
  // When backend is undefined (config still loading), show every section
  // rather than hiding the whole form.
  if (backend === undefined) return true
  return allowed.includes(backend)
}

interface ConfigFormProps {
  value: ConfigFormValue
  onChange: (next: ConfigFormValue) => void
  errors?: ValidationFieldError[]
  /**
   * When true the form is rendered for browsing only — every input is
   * disabled (via a wrapping <fieldset disabled> plus a context flag for the
   * widgets that don't naturally inherit it). Used by ConfigPreview so the
   * config is rendered as a structured overview instead of raw YAML.
   */
  readOnly?: boolean
}

export function ConfigForm({ value, onChange, errors, readOnly = false }: ConfigFormProps) {
  const errorMap = useMemo(() => buildErrorMap(errors), [errors])

  // Stable updater factory to keep child callbacks identity-stable when
  // value/onChange themselves are stable.
  const set = useCallback(
    (path: ReadonlyArray<string | number>, next: unknown) => {
      onChange(setIn(value, path, next))
    },
    [value, onChange],
  )

  const backendType = (value.backend?.type ?? undefined) as
    | "kohya"
    | "diffusion-pipe"
    | "anima_lora"
    | undefined
  const arch = value.baseModel?.arch ?? ""
  // ArchPaths section is collapsed by default but auto-expands for arches
  // that almost always need a per-component path filled in.
  const archPathsAutoOpen =
    arch === "flux" ||
    arch === "flux2" ||
    arch === "sd3" ||
    arch === "anima" ||
    arch === "hunyuan_image"
  const flowMatchVisible = FLOW_MATCH_ARCHES.has(arch)

  const body = (
    <div className="space-y-3">
      <Section
        icon={<Wand2 className="size-3.5" />}
        title="后端选择"
        subtitle="先决定训练后端 — kohya / diffusion-pipe / anima_lora。下方表单按所选后端动态显示。"
        defaultOpen
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
            value={value.backend?.diffusionPipe}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      {value.backend?.type === "anima_lora" && (
        <Section
          icon={<Sparkles className="size-3.5" />}
          title="anima_lora 选项"
          subtitle="method / preset 与上游 lora.toml 对齐;turbo 字段切到 distill 路径"
          defaultOpen
        >
          <BackendAnimaLoraFields
            value={value.backend?.animaLora}
            optimizer={value.optimizer}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      <Section
        icon={<Cpu className="size-3.5" />}
        title="基础模型"
        subtitle="选择架构与待微调的 .safetensors 检查点"
        defaultOpen
      >
        <BaseModelFields
          value={value.baseModel}
          set={set}
          errorMap={errorMap}
          backendType={backendType}
        />
      </Section>

      <Section
        icon={<Sparkles className="size-3.5" />}
        title="架构组件路径"
        subtitle="FLUX / SD3 / Anima 等多文件 bundle 的逐组件路径"
        defaultOpen={archPathsAutoOpen}
      >
        <ArchPathsFields
          value={value.baseModel?.archPaths}
          set={set}
          errorMap={errorMap}
          arch={arch}
        />
      </Section>

      <Section
        icon={<FileImage className="size-3.5" />}
        title="数据集"
        subtitle="训练图片的位置与加载方式"
        defaultOpen
      >
        <DatasetFields value={value.dataset} set={set} errorMap={errorMap} />
      </Section>

      {showsForBackend("network", backendType) && (
        <Section
          icon={<Layers className="size-3.5" />}
          title="网络"
          subtitle="LoRA 结构：rank、alpha、目标模块"
          defaultOpen
        >
          <NetworkFields value={value.network} set={set} errorMap={errorMap} />
        </Section>
      )}

      {showsForBackend("optimizer", backendType) && (
        <Section
          icon={<SlidersHorizontal className="size-3.5" />}
          title="优化器与学习率"
          subtitle="权重的更新策略"
        >
          <OptimizerFields value={value.optimizer} set={set} errorMap={errorMap} />
        </Section>
      )}

      {showsForBackend("loss", backendType) && (
        <Section
          icon={<Activity className="size-3.5" />}
          title="损失整形"
          subtitle="Min-SNR、噪声偏移、loss_type 等"
        >
          <LossFields value={value.loss} set={set} errorMap={errorMap} />
        </Section>
      )}

      {showsForBackend("advancedLoss", backendType) && (
        <Section
          icon={<Flame className="size-3.5" />}
          title="高级损失"
          subtitle="multires noise / huber schedule / pseudo huber / v_pred_like"
        >
          <AdvancedLossFields value={value.loss} set={set} errorMap={errorMap} />
        </Section>
      )}

      {flowMatchVisible && showsForBackend("flowMatch", backendType) && (
        <Section
          icon={<Shuffle className="size-3.5" />}
          title="Flow Matching"
          subtitle="FLUX / SD3 / Lumina / Anima / HunyuanImage / chroma 专用"
        >
          <FlowMatchFields
            value={value.flowMatch}
            set={set}
            errorMap={errorMap}
            arch={arch}
          />
        </Section>
      )}

      <Section
        icon={<Settings2 className="size-3.5" />}
        title="训练计划"
        subtitle="回合数、批大小、梯度累积"
        defaultOpen
      >
        <ScheduleFields value={value.schedule} set={set} errorMap={errorMap} />
      </Section>

      {showsForBackend("attention", backendType) && (
        <Section
          icon={<Zap className="size-3.5" />}
          title="注意力内核"
          subtitle="Flash / xformers / sdpa；按 GPU 计算能力自动门控"
        >
          <AttentionFields value={value.attention} set={set} errorMap={errorMap} />
        </Section>
      )}

      {showsForBackend("precision", backendType) && (
        <Section
          icon={<PaintBucket className="size-3.5" />}
          title="精度与显存"
          subtitle="混合精度、梯度检查点、潜变量缓存"
        >
          <PrecisionFields value={value} set={set} errorMap={errorMap} />
        </Section>
      )}

      {showsForBackend("optimization", backendType) && (
        <Section
          icon={<Rocket className="size-3.5" />}
          title="训练优化"
          subtitle="torch_compile / full_bf16 / blocks_to_swap / fp8 等显存与速度开关"
        >
          <OptimizationFields
            value={value.optimization}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      {showsForBackend("dataloader", backendType) && (
        <Section
          icon={<Database className="size-3.5" />}
          title="DataLoader"
          subtitle="num_workers / vae_batch_size / 缓存批大小"
        >
          <DataLoaderFields
            value={value.dataloader}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      {showsForBackend("augmentation", backendType) && (
        <Section
          icon={<Shuffle className="size-3.5" />}
          title="数据增强"
          subtitle="flip / 颜色 / 随机裁剪 / face crop / alpha mask（kohya）"
        >
          <AugmentationFields
            value={value.augmentation}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      {showsForBackend("validation", backendType) && (
        <Section
          icon={<Gauge className="size-3.5" />}
          title="验证"
          subtitle="留出比例与验证频率"
        >
          <ValidationFields value={value} set={set} errorMap={errorMap} />
        </Section>
      )}

      <Section
        icon={<Image className="size-3.5" />}
        title="采样预览"
        subtitle="训练过程中定期生成预览图"
      >
        <SamplingFields value={value.sampling} set={set} errorMap={errorMap} />
      </Section>

      {showsForBackend("monitoring", backendType) && (
        <Section
          icon={<LineChart className="size-3.5" />}
          title="实验跟踪"
          subtitle="把训练指标推到 wandb.ai 或自托管 W&amp;B Server"
        >
          <MonitoringFields
            value={value.monitoring}
            set={set}
            errorMap={errorMap}
          />
        </Section>
      )}

      <Section
        icon={<Folder className="size-3.5" />}
        title="输出"
        subtitle="文件名、保存频率、保存精度"
        defaultOpen
      >
        <OutputFields value={value.output} set={set} errorMap={errorMap} />
      </Section>

      {showsForBackend("resume", backendType) && (
        <Section
          icon={<History className="size-3.5" />}
          title="断点续训"
          subtitle="optimizer / scheduler state 落盘策略"
        >
          <ResumeFields value={value.resume} set={set} errorMap={errorMap} />
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
