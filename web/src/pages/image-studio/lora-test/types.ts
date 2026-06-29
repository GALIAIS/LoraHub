/**
 * LoRA 测试台共享类型与常量。
 *
 * 与具体组件解耦的纯数据定义集中在此，方便 panel / helper 复用。
 */

export interface ResultImage {
  path: string
  seed: number
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  cfg: number
  sampler: string
  lora_weight: number
  loras?: Array<{
    job_id: string
    checkpoint_path: string
    checkpoint_name: string
    weight: number
  }>
  checkpoint_path: string
  job_id: string
  x_label?: string | null
  y_label?: string | null
}

export interface LoraRow {
  id: string
  jobId: string
  checkpointPath: string
  weight: number
}

export const SIZE_PRESETS = [
  { label: "896 x 1632", width: 896, height: 1632 },
  { label: "768 x 1344", width: 768, height: 1344 },
  { label: "832 x 1216", width: 832, height: 1216 },
  { label: "1024 x 1024", width: 1024, height: 1024 },
] as const

export const AXIS_FIELDS = [
  { value: "variant", label: "Base/LoRA" },
  { value: "prompt", label: "Prompt" },
  { value: "negative_prompt", label: "Negative" },
  { value: "seed", label: "Seed" },
  { value: "lora_weight", label: "LoRA 权重" },
  { value: "cfg", label: "CFG" },
  { value: "steps", label: "Steps" },
  { value: "sampler", label: "Sampler" },
  { value: "size", label: "尺寸" },
  { value: "checkpoint", label: "Checkpoint" },
] as const

export const NEGATIVE_STRESS_VALUES = [
  "empty",
  "low quality, worst quality, blurry, bad anatomy",
  "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, watermark, text",
]

export const QUALITY_NEGATIVE =
  "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, watermark, text"
