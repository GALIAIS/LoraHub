export const METHOD_OPTIONS = [
  { value: "lora", label: "LoRA · 默认堆叠 (LoRA + OrthoLoRA + T-LoRA)" },
  { value: "postfix", label: "Postfix · 自由参数 / 条件正交后缀" },
  { value: "chimera", label: "ChimeraHydra · 双池路由 MoE" },
  { value: "easycontrol", label: "EasyControl · 自注意力图像条件" },
  { value: "full_finetune", label: "全量微调 · 训练完整 Anima DiT" },
  {
    value: "ip_adapter",
    label: "IP-Adapter · 图像交叉注意力 (PE-Core encoder)",
  },
] as const

export const PRESET_OPTIONS = [
  { value: "default", label: "default · 标准 24 GB" },
  { value: "low_vram", label: "low_vram · 8 GB (grad ckpt + unsloth offload)" },
  { value: "graft", label: "graft · blocks_to_swap = 20" },
  { value: "half", label: "half · 50 % 数据 (实验)" },
  { value: "quarter", label: "quarter · 25 % 数据" },
  { value: "tenth", label: "tenth · 10 % 数据" },
  { value: "debug", label: "debug · 0.1 % 数据 (管线打通)" },
] as const

export const TIMESTEP_OPTIONS = [
  { value: "sigmoid", label: "sigmoid · 默认" },
  { value: "uniform", label: "uniform" },
  { value: "logit_normal", label: "logit-normal" },
] as const

export const WEIGHTING_SCHEME_OPTIONS = [
  { value: "", label: "关闭 · 等权 RF 损失 (默认)" },
  {
    value: "min_snr_rf",
    label: "Min-SNR-γ · 整流流变体 (需 min_snr_gamma)",
  },
  { value: "sigma_sqrt", label: "Sigma-Sqrt" },
  { value: "logit_normal", label: "Logit-Normal" },
  { value: "mode", label: "Mode" },
  { value: "cosmap", label: "CosMap" },
] as const

export const ATTN_OPTIONS = [
  { value: "flash", label: "FlashAttention · 默认 (需 flash-attn)" },
  { value: "torch", label: "Torch SDPA · 无 flash-attn 时备选" },
  { value: "flex", label: "FlexAttention" },
  { value: "sageattn", label: "SageAttention · 仅推理" },
  { value: "xformers", label: "xFormers" },
] as const

export const COMPILE_MODE_OPTIONS = [
  { value: "", label: "关闭 · 默认" },
  { value: "blocks", label: "blocks · 分块编译 (可与 grad ckpt 共存)" },
  {
    value: "full",
    label: "full · 全图编译 (与 grad ckpt / blocks_to_swap 互斥)",
  },
] as const

export const COMPILE_INDUCTOR_OPTIONS = [
  { value: "", label: "默认" },
  { value: "default", label: "default" },
  { value: "reduce-overhead", label: "reduce-overhead · 推荐" },
  { value: "max-autotune", label: "max-autotune" },
] as const

export const MIXED_PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16 · 默认" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

export const OPTIMIZER_OPTIONS = [
  { value: "AdamW", label: "AdamW · 默认" },
  { value: "AdamW8bit", label: "AdamW8bit" },
  { value: "Lion", label: "Lion" },
  { value: "Prodigy", label: "Prodigy" },
  { value: "CAME", label: "CAME · 显存友好二阶矩 (LyCORIS / 风格 LoRA 推荐)" },
] as const

export const LR_SCHEDULER_OPTIONS = [
  { value: "constant", label: "constant · 默认" },
  { value: "cosine", label: "cosine" },
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "linear", label: "linear" },
  { value: "polynomial", label: "polynomial" },
] as const

export const BUCKET_TABLE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "1536", label: "1536² native (9216+9240)" },
] as const
