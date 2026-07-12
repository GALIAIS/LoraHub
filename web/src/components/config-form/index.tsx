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
import { BackendAiToolkitFields } from "./sections/backend-ai-toolkit"
import { BackendDiffusionPipeFields } from "./sections/backend-diffusion-pipe"
import { ValidationFields } from "./sections/validation"
import { ResumeFields } from "./sections/resume"
import { FLOW_MATCH_ARCHES } from "./options"
import { defaultArchFor, isArchSupported } from "./backend-meta"
import { buildErrorMap, setIn } from "./types"
import type { BackendKey, ConfigFormValue } from "./types"
import { ReadOnlyProvider, Section } from "./widgets"

export type { ConfigFormValue } from "./types"

export type { BackendKey } from "./types"
const AI_TOOLKIT_DEFAULT_CHECKPOINT = "krea/Krea-2-Raw"
const AI_TOOLKIT_KREA_CHECKPOINTS = new Set([
  "krea/Krea-2-Raw",
  "krea/Krea-2-Turbo",
])
const AI_TOOLKIT_NETWORK_TYPES = new Set(["lora", "dora", "loha", "lokr", "lorm"])

const UNUSED_SECTIONS: Record<BackendKey, readonly string[]> = {
  kohya: [],
  "diffusion-pipe": [],
  anima_lora: [
    "network",
    "loss",
    "advancedLoss",
    "flowMatch",
    "attention",
    "precision",
    "optimization",
    "dataloader",
    "augmentation",
    "validation",
    "output",
  ],
  ai_toolkit: [
    "archPaths",
    "loss",
    "advancedLoss",
    "flowMatch",
    "attention",
    "dataloader",
    "augmentation",
    "validation",
    "resume",
    "monitoring",
  ],
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
  return [...UNUSED_SECTIONS[backend]]
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

  const setBackendType = useCallback(
    (raw: string) => {
      const nextBackend = raw as BackendKey
      let next = setIn(value, ["backend", "type"], nextBackend)
      let archChanged = false
      if (!isArchSupported(nextBackend, value.baseModel?.arch)) {
        const nextArch = defaultArchFor(nextBackend)
        next = setIn(next, ["baseModel", "arch"], nextArch)
        archChanged = true
        if (nextArch !== "sdxl" && next.baseModel?.archVariant) {
          next = setIn(next, ["baseModel", "archVariant"], "")
        }
      }
      if (nextBackend === "ai_toolkit") {
        const checkpoint = String(next.baseModel?.checkpoint ?? "").trim()
        if (archChanged || !checkpoint || !AI_TOOLKIT_KREA_CHECKPOINTS.has(checkpoint)) {
          next = setIn(next, ["baseModel", "checkpoint"], AI_TOOLKIT_DEFAULT_CHECKPOINT)
        }
        if (!AI_TOOLKIT_NETWORK_TYPES.has(String(next.network?.type ?? "lora"))) {
          next = setIn(next, ["network", "type"], "lora")
          next = setIn(next, ["network", "convDim"], null)
          next = setIn(next, ["network", "convAlpha"], null)
        }
        next = setIn(next, ["backend", "gpuDispatch"], {
          mode: "one-job-per-gpu",
        })
      }
      onChange(next)
    },
    [value, onChange],
  )

  const backendType = (value.backend?.type ?? undefined) as
    | "kohya"
    | "diffusion-pipe"
    | "anima_lora"
    | "ai_toolkit"
    | undefined
  const arch = value.baseModel?.arch ?? ""
  const flowMatchVisible = FLOW_MATCH_ARCHES.has(arch)

  const body = (
    <div className="space-y-3">
      <Section
        icon={<Wand2 className="size-3.5" />}
        title="后端选择"
        subtitle="选择训练后端。表单按后端显示对应字段。"
      >
        <BackendFields
          value={value.backend}
          set={set}
          errorMap={errorMap}
          onTypeChange={setBackendType}
        />
      </Section>

      {backendType === "diffusion-pipe" ? (
        <DiffusionPipeForm
          value={value}
          set={set}
          errorMap={errorMap}
          arch={arch}
          flowMatchVisible={flowMatchVisible}
        />
      ) : backendType === "anima_lora" ? (
        <AnimaLoraForm value={value} set={set} errorMap={errorMap} arch={arch} />
      ) : backendType === "ai_toolkit" ? (
        <AiToolkitForm value={value} set={set} errorMap={errorMap} />
      ) : (
        <KohyaForm
          value={value}
          set={set}
          errorMap={errorMap}
          arch={arch}
          flowMatchVisible={flowMatchVisible}
        />
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

type BackendFormProps = {
  value: ConfigFormValue
  set: (path: ReadonlyArray<string | number>, next: unknown) => void
  errorMap: ReturnType<typeof buildErrorMap>
  arch?: string
  flowMatchVisible?: boolean
}

function KohyaForm({ value, set, errorMap, arch = "", flowMatchVisible = false }: BackendFormProps) {
  return (
    <>
      <BaseModelSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <ArchPathsSection value={value} set={set} errorMap={errorMap} arch={arch} backendType="kohya" />
      <DatasetSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <NetworkSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <OptimizerSection value={value} set={set} errorMap={errorMap} />
      <LossSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <AdvancedLossSection value={value} set={set} errorMap={errorMap} />
      {flowMatchVisible && (
        <FlowMatchSection value={value} set={set} errorMap={errorMap} arch={arch} />
      )}
      <ScheduleSection value={value} set={set} errorMap={errorMap} />
      <AttentionSection value={value} set={set} errorMap={errorMap} />
      <PrecisionSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <OptimizationSection value={value} set={set} errorMap={errorMap} />
      <DataLoaderSection value={value} set={set} errorMap={errorMap} />
      <AugmentationSection value={value} set={set} errorMap={errorMap} />
      <ValidationSection value={value} set={set} errorMap={errorMap} />
      <SamplingSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <MonitoringSection value={value} set={set} errorMap={errorMap} />
      <OutputSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
      <ResumeSection value={value} set={set} errorMap={errorMap} backendType="kohya" />
    </>
  )
}

function DiffusionPipeForm({ value, set, errorMap, arch = "" }: BackendFormProps) {
  return (
    <>
      <Section icon={<Workflow className="size-3.5" />} title="diffusion-pipe 选项" subtitle="diffusion-pipe 后端配置">
        <BackendDiffusionPipeFields value={value.backend?.diffusionPipe} set={set} errorMap={errorMap} />
      </Section>
      <BaseModelSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <ArchPathsSection value={value} set={set} errorMap={errorMap} arch={arch} backendType="diffusion-pipe" />
      <DatasetSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <NetworkSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <OptimizerSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <LossSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <ScheduleSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <PrecisionSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <OptimizationSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <DataLoaderSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <SamplingSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <OutputSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <ResumeSection value={value} set={set} errorMap={errorMap} backendType="diffusion-pipe" />
      <MonitoringSection value={value} set={set} errorMap={errorMap} />
    </>
  )
}

function AnimaLoraForm({ value, set, errorMap, arch = "" }: BackendFormProps) {
  return (
    <>
      <Section icon={<Sparkles className="size-3.5" />} title="anima_lora 选项" subtitle="anima_lora 后端配置">
        <BackendAnimaLoraFields
          value={value.backend?.animaLora}
          optimizer={value.optimizer}
          set={set}
          errorMap={errorMap}
        />
      </Section>
      <BaseModelSection value={value} set={set} errorMap={errorMap} backendType="anima_lora" />
      <ArchPathsSection value={value} set={set} errorMap={errorMap} arch={arch} backendType="anima_lora" />
      <DatasetSection value={value} set={set} errorMap={errorMap} backendType="anima_lora" />
      <ScheduleSection value={value} set={set} errorMap={errorMap} backendType="anima_lora" />
      <SamplingSection value={value} set={set} errorMap={errorMap} backendType="anima_lora" />
      <ResumeSection value={value} set={set} errorMap={errorMap} backendType="anima_lora" />
      <MonitoringSection value={value} set={set} errorMap={errorMap} />
    </>
  )
}

function AiToolkitForm({ value, set, errorMap }: BackendFormProps) {
  return <BackendAiToolkitFields value={value} set={set} errorMap={errorMap} />
}

function BaseModelSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Cpu className="size-3.5" />} title="基础模型" subtitle="模型架构与检查点">
      <BaseModelFields value={value.baseModel} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function ArchPathsSection({ value, set, errorMap, arch = "", backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Sparkles className="size-3.5" />} title="架构组件路径" subtitle="多文件模型组件路径">
      <ArchPathsFields
        value={value.baseModel?.archPaths}
        set={set}
        errorMap={errorMap}
        arch={arch}
        backendType={backendType}
      />
    </Section>
  )
}

function DatasetSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<FileImage className="size-3.5" />} title="数据集" subtitle={`${backendType} 数据集配置`}>
      <DatasetFields value={value.dataset} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function NetworkSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Layers className="size-3.5" />} title="网络" subtitle="LoRA 结构">
      <NetworkFields value={value.network} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function OptimizerSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType?: BackendKey }) {
  return (
    <Section icon={<SlidersHorizontal className="size-3.5" />} title="优化器与学习率" subtitle="权重更新策略">
      <OptimizerFields value={value.optimizer} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function LossSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Activity className="size-3.5" />} title="损失整形" subtitle="后端损失参数">
      <LossFields value={value.loss} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function AdvancedLossSection({ value, set, errorMap }: BackendFormProps) {
  return (
    <Section icon={<Flame className="size-3.5" />} title="高级损失" subtitle="kohya 高级 loss 参数">
      <AdvancedLossFields value={value.loss} set={set} errorMap={errorMap} />
    </Section>
  )
}

function FlowMatchSection({ value, set, errorMap, arch = "" }: BackendFormProps) {
  return (
    <Section icon={<Shuffle className="size-3.5" />} title="Flow Matching" subtitle="flow-matching 参数">
      <FlowMatchFields value={value.flowMatch} set={set} errorMap={errorMap} arch={arch} />
    </Section>
  )
}

function ScheduleSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType?: BackendKey }) {
  return (
    <Section icon={<Settings2 className="size-3.5" />} title="训练计划" subtitle="回合数、批大小、梯度累积">
      <ScheduleFields value={value.schedule} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function AttentionSection({ value, set, errorMap }: BackendFormProps) {
  return (
    <Section icon={<Zap className="size-3.5" />} title="注意力内核" subtitle="注意力实现选择">
      <AttentionFields value={value.attention} set={set} errorMap={errorMap} />
    </Section>
  )
}

function PrecisionSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<PaintBucket className="size-3.5" />} title="精度与显存" subtitle="精度、检查点与缓存">
      <PrecisionFields value={value} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function OptimizationSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType?: BackendKey }) {
  return (
    <Section icon={<Rocket className="size-3.5" />} title="训练优化" subtitle="速度与显存开关">
      <OptimizationFields value={value.optimization} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function DataLoaderSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType?: BackendKey }) {
  return (
    <Section icon={<Database className="size-3.5" />} title="DataLoader" subtitle="加载与缓存批大小">
      <DataLoaderFields value={value.dataloader} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function AugmentationSection({ value, set, errorMap }: BackendFormProps) {
  return (
    <Section icon={<Shuffle className="size-3.5" />} title="数据增强" subtitle="kohya 数据增强">
      <AugmentationFields value={value.augmentation} set={set} errorMap={errorMap} />
    </Section>
  )
}

function ValidationSection({ value, set, errorMap }: BackendFormProps) {
  return (
    <Section icon={<Gauge className="size-3.5" />} title="验证" subtitle="验证集与验证频率">
      <ValidationFields value={value} set={set} errorMap={errorMap} />
    </Section>
  )
}

function SamplingSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Image className="size-3.5" />} title="采样预览" subtitle="训练预览图">
      <SamplingFields value={value.sampling} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function MonitoringSection({ value, set, errorMap }: BackendFormProps) {
  return (
    <Section icon={<LineChart className="size-3.5" />} title="实验跟踪" subtitle="Weights & Biases">
      <MonitoringFields value={value.monitoring} set={set} errorMap={errorMap} />
    </Section>
  )
}

function OutputSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<Folder className="size-3.5" />} title="输出" subtitle="文件名、保存频率、保存精度">
      <OutputFields value={value.output} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}

function ResumeSection({ value, set, errorMap, backendType }: BackendFormProps & { backendType: BackendKey }) {
  return (
    <Section icon={<History className="size-3.5" />} title="断点续训" subtitle="恢复与 state 保存">
      <ResumeFields value={value.resume} set={set} errorMap={errorMap} backendType={backendType} />
    </Section>
  )
}
