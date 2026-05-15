/**
 * Module-level option lists for select widgets.
 *
 * Hoisted so they're allocated once (rerender-defer-reads, js-cache-storage).
 */

export const ARCH_OPTIONS = [
  { value: "sdxl", label: "SDXL (1024)" },
  { value: "sd15", label: "SD 1.5 (512/768)" },
  { value: "flux", label: "Flux" },
  { value: "sd3", label: "SD 3" },
] as const

export const NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
  { value: "locon", label: "LoCon" },
  { value: "loha", label: "LoHA" },
  { value: "dora", label: "DoRA" },
] as const

export const OPTIMIZER_OPTIONS = [
  { value: "adamw8bit", label: "AdamW 8bit (bitsandbytes)" },
  { value: "adamw", label: "AdamW" },
  { value: "lion", label: "Lion" },
  { value: "lion8bit", label: "Lion 8bit" },
  { value: "prodigy", label: "Prodigy" },
  { value: "dadaptation", label: "D-Adaptation" },
] as const

export const LR_SCHEDULE_OPTIONS = [
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "cosine", label: "cosine" },
  { value: "linear", label: "linear" },
  { value: "constant", label: "constant" },
  { value: "constant_with_warmup", label: "constant_with_warmup" },
  { value: "polynomial", label: "polynomial" },
] as const

export const PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16 (Ampere+)" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

export const SAVE_DTYPE_OPTIONS = [
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "float", label: "float32" },
] as const

export const CAPTION_STRATEGY_OPTIONS = [
  { value: "tag_file", label: "tag_file (.txt next to images)" },
  { value: "filename", label: "filename" },
  { value: "none", label: "none" },
] as const

export const BACKEND_OPTIONS = [
  { value: "kohya", label: "kohya-ss/sd-scripts" },
  { value: "diffusers", label: "🤗 diffusers (planned)" },
] as const
