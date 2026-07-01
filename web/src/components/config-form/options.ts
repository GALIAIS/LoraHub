/**
 * Module-level option lists for select widgets.
 *
 * Hoisted so they're allocated once (rerender-defer-reads, js-cache-storage).
 */

// Mirrors BaseModelConfig.arch in lorahub/core/config/schema.py.
// Order: 常用静态图 → 其他主流静态图 → 动漫子族 → 视频 → 其它 / 实验性。
export const ARCH_OPTIONS = [
  // 常用静态图
  { value: "sdxl", label: "SDXL · Stable Diffusion XL (1024)" },
  { value: "sd15", label: "SD 1.5 · Stable Diffusion 1.5 (512/768)" },
  { value: "sd3", label: "SD 3 · Stable Diffusion 3" },
  { value: "flux", label: "FLUX.1 · black-forest-labs" },
  { value: "krea2", label: "Krea 2 · krea/Krea-2-Raw" },
  // 其他主流静态图
  { value: "sd2", label: "SD 2 · Stable Diffusion 2.x" },
  { value: "flux2", label: "FLUX 2 · black-forest-labs (next-gen)" },
  { value: "qwen_image", label: "Qwen-Image · 阿里通义" },
  { value: "hidream", label: "HiDream · I1 系列" },
  // 动漫子族
  { value: "anima", label: "Anima · circlestone-labs" },
  { value: "lumina", label: "Lumina · Lumina-Next" },
  { value: "hunyuan_image", label: "HunyuanImage · 腾讯混元图像" },
  // 视频
  { value: "hunyuan_video", label: "HunyuanVideo · 腾讯混元视频" },
  { value: "hunyuan_video_15", label: "HunyuanVideo 1.5 · 腾讯混元视频 v1.5" },
  { value: "wan", label: "Wan2.1 / Wan2.2 · 阿里万相 (视频)" },
  { value: "ltx_video", label: "LTX-Video · Lightricks (视频)" },
  { value: "ltx2", label: "LTX2 · Lightricks 新代 (视频)" },
  { value: "cosmos", label: "Cosmos · NVIDIA (视频)" },
  { value: "cosmos_predict2", label: "Cosmos Predict2 · NVIDIA (视频)" },
  // 其它 / 实验性
  { value: "chroma", label: "Chroma · lodestones" },
  { value: "omnigen2", label: "OmniGen2 · VectorSpaceLab" },
  { value: "auraflow", label: "AuraFlow · fal.ai" },
  { value: "z_image", label: "Z-Image · zentropy-ai" },
  { value: "ernie_image", label: "ERNIE Image · 百度文心" },
] as const

// SDXL sub-architectures sharing the SDXL backbone but trained on different
// finetune lineages. Empty string == 「无」(plain SDXL, no variant).
export const ARCH_VARIANT_OPTIONS = [
  { value: "", label: "无 · 标准 SDXL" },
  { value: "pony", label: "Pony Diffusion v6" },
  { value: "illustrious", label: "Illustrious XL" },
  { value: "noobai", label: "NoobAI XL" },
  { value: "animagine", label: "Animagine XL" },
] as const

export const NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
  { value: "locon", label: "LoCon" },
  { value: "loha", label: "LoHA" },
  { value: "lokr", label: "LoKr" },
  { value: "lorm", label: "LoRM" },
  { value: "dora", label: "DoRA" },
] as const

export const AI_TOOLKIT_NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
  { value: "dora", label: "DoRA" },
  { value: "loha", label: "LoHA" },
  { value: "lokr", label: "LoKr" },
  { value: "lorm", label: "LoRM" },
] as const

export const KOHYA_NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
  { value: "locon", label: "LoCon" },
  { value: "loha", label: "LoHA" },
  { value: "lokr", label: "LoKr" },
  { value: "dora", label: "DoRA" },
] as const

export const DIFFUSION_PIPE_NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
] as const

export const OPTIMIZER_OPTIONS = [
  { value: "adamw8bit", label: "AdamW8bit · bitsandbytes" },
  { value: "adamw", label: "AdamW" },
  { value: "lion", label: "Lion" },
  { value: "lion8bit", label: "Lion8bit" },
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
  { value: "bf16", label: "bf16 · Ampere+" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

export const SAVE_DTYPE_OPTIONS = [
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "float", label: "float32" },
] as const

export const CAPTION_STRATEGY_OPTIONS = [
  { value: "tag_file", label: "tag_file · 同名 .txt 描述文件" },
  { value: "filename", label: "filename · 文件名作为描述" },
  { value: "none", label: "none · 不使用描述" },
] as const

export const BACKEND_OPTIONS = [
  { value: "kohya", label: "kohya-ss / sd-scripts" },
  { value: "diffusion-pipe", label: "tdrussell / diffusion-pipe" },
  { value: "anima_lora", label: "sorryhyun / anima_lora · vendored" },
  { value: "ai_toolkit", label: "ostris / ai-toolkit · vendored" },
] as const

export const LOSS_TYPE_OPTIONS = [
  { value: "l2", label: "L2 · MSE 默认" },
  { value: "huber", label: "Huber · 鲁棒" },
  { value: "smooth_l1", label: "Smooth L1" },
] as const

export const PARTITION_METHOD_OPTIONS = [
  { value: "parameters", label: "parameters · 按参数数均分" },
  { value: "uniform", label: "uniform · 按层数均分" },
  { value: "type:transformer_layer", label: "type:transformer_layer" },
] as const

// FlowMatchConfig.timestep_sampling
export const FLOW_MATCH_TIMESTEP_OPTIONS = [
  { value: "", label: "默认" },
  { value: "logit_normal", label: "logit_normal" },
  { value: "uniform", label: "uniform" },
  { value: "sigma_uniform", label: "sigma_uniform" },
  { value: "mode", label: "mode" },
  { value: "cosmap", label: "cosmap" },
] as const

export const FLOW_MATCH_PRED_TYPE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "raw", label: "raw" },
  { value: "additive", label: "additive" },
  { value: "sigma_scaled", label: "sigma_scaled" },
] as const

export const FLOW_MATCH_WEIGHTING_OPTIONS = [
  { value: "", label: "默认" },
  { value: "sigma_sqrt", label: "sigma_sqrt" },
  { value: "logit_normal", label: "logit_normal" },
  { value: "mode", label: "mode" },
  { value: "cosmap", label: "cosmap" },
  { value: "none", label: "none" },
] as const

// LossConfig.huber_schedule
export const HUBER_SCHEDULE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "constant", label: "constant" },
  { value: "exponential", label: "exponential" },
  { value: "snr", label: "snr" },
] as const

// BucketConfig.resize_interpolation
export const RESIZE_INTERPOLATION_OPTIONS = [
  { value: "", label: "默认 · 由训练器决定" },
  { value: "lanczos", label: "Lanczos" },
  { value: "bicubic", label: "Bicubic" },
  { value: "bilinear", label: "Bilinear" },
  { value: "box", label: "Box" },
  { value: "nearest", label: "Nearest" },
  { value: "hamming", label: "Hamming" },
] as const

// CaptionConfig.max_token_length
export const MAX_TOKEN_LENGTH_OPTIONS = [
  { value: "", label: "默认 · 75" },
  { value: "75", label: "75" },
  { value: "150", label: "150" },
  { value: "225", label: "225" },
] as const

// NetworkConfig.dtype (LoRA training dtype on dp)
export const NETWORK_DTYPE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "fp32", label: "fp32" },
] as const

// ArchPathsConfig.t5xxl_dtype
export const T5_DTYPE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "fp32", label: "fp32" },
  { value: "fp8", label: "fp8" },
] as const

// DiffusionPipeOptions.transformer_dtype
export const DP_TRANSFORMER_DTYPE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float16", label: "float16" },
  { value: "float8_e4m3fn", label: "float8_e4m3fn" },
  { value: "float8_e5m2", label: "float8_e5m2" },
] as const

// DiffusionPipeOptions.diffusion_model_dtype
export const DP_DIFFUSION_DTYPE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float16", label: "float16" },
  { value: "float8_e4m3fn", label: "float8_e4m3fn" },
] as const

// DiffusionPipeOptions.timestep_sample_method
export const DP_TIMESTEP_SAMPLE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "logit_normal", label: "logit_normal" },
  { value: "uniform", label: "uniform" },
] as const

// DiffusionPipeOptions.video_clip_mode
export const DP_VIDEO_CLIP_MODE_OPTIONS = [
  { value: "single_beginning", label: "single_beginning" },
  { value: "single_middle", label: "single_middle" },
] as const

// Architectures that consume FlowMatchConfig knobs.
// Anything else → flow_match section is hidden.
export const FLOW_MATCH_ARCHES: ReadonlySet<string> = new Set([
  "flux",
  "flux2",
  "sd3",
  "lumina",
  "anima",
  "hunyuan_image",
  "chroma",
  "qwen_image",
])
