/**
 * Friendly Chinese labels for the dotted field paths the backend
 * uses on validation issues.
 *
 * Pydantic / backend.validate emit raw paths like
 * ``backend.repo_path`` or ``base_model.checkpoint``. Showing those
 * verbatim in toasts / preflight panels is fine for power users but
 * confusing for first-time form users — they can't trace which
 * input triggered the warning. This helper turns the path into:
 *
 *   1. A friendly Chinese label when we have one in the dictionary
 *      (e.g. ``backend.repo_path`` → ``后端 / 训练库路径``).
 *   2. A "pretty" tail when we don't (e.g. ``backend.foo.bar`` →
 *      ``backend / foo / bar``).
 *
 * The original raw path is also returned (and rendered in monospace
 * next to the label) so power users can still copy / search it. The
 * goal is "visible identifier + readable label", not replacement.
 */

const RAW_PATH_LABELS: Record<string, string> = {
  // Top-level config
  "schemaVersion": "Schema 版本",

  // baseModel.*
  "base_model.arch": "底模 / 架构",
  "baseModel.arch": "底模 / 架构",
  "base_model.checkpoint": "底模 / checkpoint 路径",
  "baseModel.checkpoint": "底模 / checkpoint 路径",
  "base_model.archPaths.qwen3": "底模 / qwen3 文本编码器路径",
  "baseModel.archPaths.qwen3": "底模 / qwen3 文本编码器路径",
  "base_model.archPaths.ae": "底模 / VAE 路径",
  "baseModel.archPaths.ae": "底模 / VAE 路径",
  "base_model.t5xxlDtype": "底模 / T5 文本编码器精度",
  "baseModel.t5xxlDtype": "底模 / T5 文本编码器精度",

  // dataset.*
  "dataset.source": "数据集 / 路径",
  "dataset.resolution": "数据集 / 分辨率",
  "dataset.bucket": "数据集 / Bucket",
  "dataset.bucket.min": "数据集 / Bucket / 最小边长",
  "dataset.bucket.max": "数据集 / Bucket / 最大边长",
  "dataset.bucket.minSize": "数据集 / Bucket / 最小边长",
  "dataset.bucket.maxSize": "数据集 / Bucket / 最大边长",
  "dataset.caption": "数据集 / 标注",
  "dataset.caption.dropRate": "数据集 / 标注 / 丢弃率",
  "dataset.caption.keepTokens": "数据集 / 标注 / 保留 token 数",
  "dataset.caption.shuffle": "数据集 / 标注 / 打乱",
  "dataset.numRepeats": "数据集 / 重复次数",

  // network.*
  "network.type": "网络 / 类型",
  "network.rank": "网络 / rank",
  "network.alpha": "网络 / alpha",

  // optimizer.*
  "optimizer.type": "优化器 / 类型",
  "optimizer.lr.unet": "优化器 / UNet 学习率",
  "optimizer.lr.textEncoder": "优化器 / 文本编码器学习率",
  "optimizer.lr.text_encoder": "优化器 / 文本编码器学习率",
  "optimizer.scheduler.type": "优化器 / 调度器",

  // schedule.*
  "schedule.epochs": "训练 / 轮数",
  "schedule.batchSize": "训练 / 单步批量",
  "schedule.batch_size": "训练 / 单步批量",
  "schedule.gradAccum": "训练 / 梯度累积",
  "schedule.grad_accum": "训练 / 梯度累积",
  "schedule.maxSteps": "训练 / 最大步数",

  // sampling.*
  "sampling.enabled": "训练采样 / 启用",
  "sampling.everyNEpochs": "训练采样 / 每 N 轮",
  "sampling.resolution": "训练采样 / 分辨率",

  // output.*
  "output.name": "输出 / 名称",
  "output.saveEveryNEpochs": "输出 / 每 N 轮保存",
  "output.save_every_n_epochs": "输出 / 每 N 轮保存",
  "output.saveDtype": "输出 / 保存精度",

  // backend.*
  "backend.type": "后端 / 类型",
  "backend.repo_path": "后端 / 训练库路径",
  "backend.repoPath": "后端 / 训练库路径",
  "backend.python_executable": "后端 / Python 解释器",
  "backend.pythonExecutable": "后端 / Python 解释器",
  "backend.extra_args": "后端 / 额外参数",
  "backend.extraArgs": "后端 / 额外参数",
  "backend.gpuDispatch.mode": "后端 / GPU 调度",
  "backend.gpuDispatch.numGpus": "后端 / GPU 数量",
  "backend.distributed.strategy": "后端 / 分布式策略",
  "backend.distributed.fsdp.shardingStrategy": "后端 / FSDP 分片",
  "backend.distributed.fsdp.autoWrapPolicy": "后端 / FSDP 包裹",
  "backend.distributed.fsdp.minNumParams": "后端 / FSDP 包裹阈值",
  "backend.distributed.fsdp.stateDictType": "后端 / FSDP 保存方式",
  "backend.distributed.fsdp.cpuOffload": "后端 / FSDP CPU offload",
  "backend.distributed.zero.stage": "后端 / ZeRO stage",
  "backend.distributed.zero.offloadOptimizer": "后端 / ZeRO 优化器 offload",
  "backend.distributed.zero.offloadParam": "后端 / ZeRO 参数 offload",
  "backend.distributed.zero.overlapComm": "后端 / ZeRO 通信重叠",

  // backend.animaLora.*
  "backend.animaLora.networkDim": "anima_lora / 网络 dim",
  "backend.animaLora.networkAlpha": "anima_lora / 网络 alpha",
  "backend.animaLora.optimizerType": "anima_lora / 优化器",
  "backend.animaLora.lrScheduler": "anima_lora / 学习率调度器",
  "backend.animaLora.learningRate": "anima_lora / 学习率",
  "backend.animaLora.maxTrainEpochs": "anima_lora / 最大训练轮数",
  "backend.animaLora.saveEveryNEpochs": "anima_lora / 每 N 轮保存",
  "backend.animaLora.checkpointingEpochs": "anima_lora / 每 N 轮 checkpoint",
  "backend.animaLora.captionDropoutRate": "anima_lora / caption 丢弃率",
  "backend.animaLora.useShuffledCaptionVariants": "anima_lora / 打乱 caption 变体",
  "backend.animaLora.cacheLatents": "anima_lora / 缓存 latents",
  "backend.animaLora.cacheTextEncoderOutputs": "anima_lora / 缓存文本编码器输出",
  "backend.animaLora.cacheLlmAdapterOutputs": "anima_lora / 缓存 LLM Adapter 输出",
  "backend.animaLora.cacheLatentsToDisk": "anima_lora / latents 落盘",
  "backend.animaLora.cacheTextEncoderOutputsToDisk": "anima_lora / TE 输出落盘",
  "backend.animaLora.staticTokenCount": "anima_lora / 静态 token 数",
  "backend.animaLora.vaeChunkSize": "anima_lora / VAE 分块大小",
  "backend.animaLora.vaeDisableCache": "anima_lora / 禁用 VAE 缓存",
  "backend.animaLora.attnMode": "anima_lora / 注意力实现",
  "backend.animaLora.useCustomDownAutograd": "anima_lora / 自定义 down autograd",
  "backend.animaLora.compileMode": "anima_lora / 编译模式",
  "backend.animaLora.compileInductorMode": "anima_lora / Inductor 模式",
  "backend.animaLora.torchCompile": "anima_lora / torch.compile",
  "backend.animaLora.mixedPrecision": "anima_lora / 混合精度",
  "backend.animaLora.blocksToSwap": "anima_lora / Block Swap 数量",
  "backend.animaLora.gradientCheckpointing": "anima_lora / 梯度 checkpoint",
  "backend.animaLora.unslothOffloadCheckpointing": "anima_lora / Unsloth Offload",
  "backend.animaLora.cpuOffloadCheckpointing": "anima_lora / CPU Offload Ckpt",
  "backend.animaLora.useCmmd": "anima_lora / CMMD 验证",
  "backend.animaLora.validationSplitNum": "anima_lora / 验证集划分数",
  "backend.animaLora.validationSeed": "anima_lora / 验证种子",
  "backend.animaLora.ema": "anima_lora / EMA",
  "backend.animaLora.nanGuard": "anima_lora / NaN 守护",
  "backend.animaLora.nanGuardRecover": "anima_lora / NaN 自动恢复",
  "backend.animaLora.nanGuardMaxConsecutive": "anima_lora / NaN 连续上限",
  "backend.animaLora.sampleGrid": "anima_lora / 样本拼图",
  "backend.animaLora.timestepSampling": "anima_lora / 时间步采样",
  "backend.animaLora.sigmoidScale": "anima_lora / sigmoid scale",
  "backend.animaLora.discreteFlowShift": "anima_lora / Flow shift",
  "backend.animaLora.weightingScheme": "anima_lora / 权重方案",
  "backend.animaLora.minSnrGamma": "anima_lora / Min-SNR γ",
  "backend.animaLora.networkTrainUnetOnly": "anima_lora / 仅训练 UNet",
  "backend.animaLora.networkModule": "anima_lora / 网络模块",
  "backend.animaLora.outputName": "anima_lora / 输出名",
  "backend.animaLora.method": "anima_lora / 方法",
  "backend.animaLora.preset": "anima_lora / 预设",
  "backend.animaLora.lora.minRank": "anima_lora / LoRA / 最小 rank",
  "backend.animaLora.lora.algorithm": "anima_lora / LoRA / 算法",
  "backend.animaLora.lora.alphaRankScale": "anima_lora / LoRA / α-rank 缩放",
  "backend.animaLora.lora.useOrtho": "anima_lora / LoRA / OrthoLoRA",
  "backend.animaLora.lora.useTimestepMask": "anima_lora / LoRA / 时间步掩码",

  // backend.diffusionPipe.*
  "backend.diffusionPipe.pipelineStages": "dp / Pipeline 阶段数",
  "backend.diffusionPipe.gradientClipping": "dp / 梯度裁剪",
  "backend.diffusionPipe.partitionMethod": "dp / 切分方法",
  "backend.diffusionPipe.partitionSplit": "dp / 切分点",
  "backend.diffusionPipe.cachingBatchSize": "dp / 缓存批量",
  "backend.diffusionPipe.blocksToSwap": "dp / Block Swap 数量",
  "backend.diffusionPipe.compile": "dp / 启用编译",
  "backend.diffusionPipe.reentrantActivationCheckpointing": "dp / 可重入激活 ckpt",
  "backend.diffusionPipe.evalEveryNEpochs": "dp / 每 N 轮评估",
  "backend.diffusionPipe.evalEveryNSteps": "dp / 每 N 步评估",
  "backend.diffusionPipe.evalEveryNExamples": "dp / 每 N 样本评估",
  "backend.diffusionPipe.evalBeforeFirstStep": "dp / 起步前评估",
  "backend.diffusionPipe.evalMicroBatchSizePerGpu": "dp / 评估 micro batch / GPU",
  "backend.diffusionPipe.imageEvalMicroBatchSizePerGpu": "dp / 图像评估 micro batch / GPU",
  "backend.diffusionPipe.transformerDtype": "dp / Transformer 精度",
  "backend.diffusionPipe.diffusionModelDtype": "dp / 扩散模型精度",
  "backend.diffusionPipe.cacheShuffleNum": "dp / 缓存打乱数",
  "backend.diffusionPipe.minAr": "dp / 最小宽高比",
  "backend.diffusionPipe.maxAr": "dp / 最大宽高比",
  "backend.diffusionPipe.uncondFraction": "dp / 无条件比例",

  // Synthetic / generic placeholders the backend uses for non-field
  // issues. Keep these short — they're surfaced in the same list.
  "@summary.vram": "汇总 / 显存预估",
  "@advisor": "智能推荐",
  "@backend": "后端校验",
  "recipe": "整体配置",
}

/**
 * Translate a backend-emitted dotted field path to a friendly label.
 *
 * Returns ``null`` when the path isn't in the dictionary so callers
 * can choose how to fall back (typically: render the raw path in
 * monospace as before, optionally suffixed with the prettified
 * "section / field" version).
 */
export function fieldLabelFor(rawPath: string): string | null {
  if (!rawPath) return null
  return RAW_PATH_LABELS[rawPath] ?? null
}

/**
 * Best-effort prettifier for paths we don't have an explicit label
 * for. Replaces dots with `` / `` so the segments breathe, leaves
 * the casing alone (camelCase / snake_case both stay readable),
 * trims any leading-`@` synthetic markers.
 */
export function prettifyFieldPath(rawPath: string): string {
  if (!rawPath) return ""
  const cleaned = rawPath.startsWith("@") ? rawPath.slice(1) : rawPath
  return cleaned.replace(/\./g, " / ")
}

/**
 * Combined renderer used by toasts / preflight / advisor panels.
 * Returns ``"<label> (<rawPath>)"`` when a friendly label exists,
 * otherwise the prettified path.
 */
export function fieldDisplay(rawPath: string): {
  label: string
  raw: string
  hasLabel: boolean
} {
  const label = fieldLabelFor(rawPath)
  if (label) return { label, raw: rawPath, hasLabel: true }
  return { label: prettifyFieldPath(rawPath), raw: rawPath, hasLabel: false }
}
