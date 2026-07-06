export const METHOD_OPTIONS = [
  { value: "lora", label: "LoRA" },
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
  { value: "default", label: "default" },
  { value: "low_vram", label: "low_vram · 8 GB (grad ckpt + unsloth offload)" },
  { value: "graft", label: "graft · blocks_to_swap = 20" },
  { value: "half", label: "half · 50 % 数据 (实验)" },
  { value: "quarter", label: "quarter · 25 % 数据" },
  { value: "tenth", label: "tenth · 10 % 数据" },
  { value: "debug", label: "debug · 0.1 % 数据" },
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
  { value: "reduce-overhead", label: "reduce-overhead" },
  { value: "max-autotune", label: "max-autotune" },
] as const

export const MIXED_PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16 · 默认" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

export const OPTIMIZER_OPTIONS = [
  { value: "AdamW", label: "AdamW · 默认" },
  { value: "AdamW8bit", label: "AdamW8bit · 省显存" },
  { value: "PagedAdamW", label: "PagedAdamW · 分页状态" },
  { value: "PagedAdamW8bit", label: "PagedAdamW8bit · 分页省显存" },
  { value: "PagedAdamW32bit", label: "PagedAdamW32bit" },
  { value: "Lion", label: "Lion · 低状态量" },
  { value: "Lion8bit", label: "Lion8bit" },
  { value: "PagedLion8bit", label: "PagedLion8bit" },
  { value: "SGDNesterov", label: "SGD Nesterov" },
  { value: "SGDNesterov8bit", label: "SGD Nesterov 8bit" },
  { value: "DAdaptation", label: "D-Adaptation" },
  { value: "DAdaptAdamPreprint", label: "DAdapt Adam Preprint" },
  { value: "DAdaptAdaGrad", label: "DAdapt AdaGrad" },
  { value: "DAdaptAdam", label: "DAdapt Adam" },
  { value: "DAdaptAdan", label: "DAdapt Adan" },
  { value: "DAdaptAdanIP", label: "DAdapt AdanIP" },
  { value: "DAdaptLion", label: "DAdapt Lion" },
  { value: "DAdaptSGD", label: "DAdapt SGD" },
  { value: "Prodigy", label: "Prodigy · 自适应 LR" },
  { value: "Adafactor", label: "Adafactor · 低显存" },
  { value: "CAME", label: "CAME · 低显存" },
  { value: "AdamWScheduleFree", label: "AdamW Schedule-Free" },
  { value: "RAdamScheduleFree", label: "RAdam Schedule-Free" },
  { value: "SGDScheduleFree", label: "SGD Schedule-Free" },
] as const

export const LR_SCHEDULER_OPTIONS = [
  { value: "constant", label: "constant · 默认" },
  { value: "constant_with_warmup", label: "constant_with_warmup" },
  { value: "linear", label: "linear" },
  { value: "cosine", label: "cosine" },
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "polynomial", label: "polynomial" },
  { value: "inverse_sqrt", label: "inverse_sqrt" },
  { value: "cosine_with_min_lr", label: "cosine_with_min_lr" },
  { value: "warmup_stable_decay", label: "warmup_stable_decay" },
  { value: "piecewise_constant", label: "piecewise_constant" },
] as const

export const TARGET_PRESET_OPTIONS = [
  { value: "all", label: "全部 Linear" },
  { value: "attention", label: "Attention" },
  { value: "cross_attention", label: "Cross-attention" },
  { value: "self_attention", label: "Self-attention" },
  { value: "mlp", label: "MLP" },
] as const

export const BUCKET_TABLE_OPTIONS = [
  { value: "", label: "默认" },
  { value: "1536", label: "1536² native (9216+9240)" },
] as const
