---
title: 配置字段参考
description: TrainingConfig schema 的每个字段,按段分组。
---

# 配置字段参考

本页给出
[`TrainingConfig`](https://github.com/GALIAIS/LoraHub/blob/main/lorahub/core/config/schema.py)
中常用字段的清单。schema 是事实源 — 下面默认值与代码同步。字段名以
**camelCase**(首选 wire form)给出,validator 也接受 snake_case。

!!! note "权威源"
    字段名、类型、默认值由 Pydantic 在 load 时校验。顶层模型设
    `extra: forbid`,未知 key 会被明确报错拒绝。

## Top-level

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `schemaVersion` | `str` | `"1.0"` | 预留给将来迁移。 |
| `precision` | `Literal["fp16", "bf16", "fp32"]` | `bf16` | 训练混合精度。 |
| `gradientCheckpointing` | `bool` | `true` | 用算力换显存。 |
| `cacheLatents` | `bool` | `true` | 第一 epoch 把 VAE latent 缓存到磁盘。 |

## `baseModel`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `arch` | `Literal[...]` | `sdxl` | 选 backend 入口。可选: `sd15`, `sd2`, `sdxl`, `sd3`, `flux`, `lumina`, `hunyuan_image`, `anima`, `flux2`, `chroma`, `hidream`, `omnigen2`, `auraflow`, `qwen_image`, `cosmos`, `cosmos_predict2`, `hunyuan_video`, `hunyuan_video_15`, `ltx_video`, `ltx2`, `wan`, `z_image`, `ernie_image`。 |
| `archVariant` | `Literal["", "pony", "illustrious", "noobai", "animagine"]` | `""` | SDXL 子风味,会微调默认值。要求 `arch == "sdxl"`。 |
| `checkpoint` | `Path` | _(必填)_ | 主干权重(`.safetensors` / `.ckpt`)。 |
| `vae` | `Path \| None` | `None` | 可选外置 VAE。 |
| `archPaths` | `ArchPathsConfig` | `{}` | 多文件架构的分件 checkpoint 路径(`clipL` / `clipG` / `t5xxl` / `ae` / `transformer` / `qwen3` / `t5Tokenizer` / `llmAdapter` 等)。 |

## `dataset`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `source` | `Path` | _(必填)_ | 图片 + `.txt` caption 所在目录。 |
| `resolution` | `list[int]` | `[1024, 1024]` | `[size]`(正方形)或 `[width, height]`。 |
| `bucket.enabled` | `bool` | `true` | 多分辨率 bucketing。 |
| `bucket.min` | `int` | `256` | 最小桶边长。 |
| `bucket.max` | `int` | `2048` | 最大桶边长。 |
| `bucket.step` | `int` | `64` | 桶尺寸量化粒度。 |
| `caption.strategy` | `Literal["tag_file", "filename", "none"]` | `tag_file` | caption 来源。**始终 snake_case** — 是 Literal,不是字段名。**UI-only**:这一字段当前不被 backend 编译器消费(B1 审计后落档)。 |
| `caption.ext` | `str` | `".txt"` | `strategy == "tag_file"` 时的文件后缀。 |
| `caption.shuffle` | `bool` | `true` | 每步随机打乱逗号分隔的 tag。 |
| `caption.dropRate` | `float` (0-1) | `0.0` | 随机丢 caption 的概率。 |
| `numRepeats` | `int (>=1)` | `1` | 每张图每 epoch 复读次数。 |

## `network`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["lora", "locon", "loha", "dora"]` | `lora` | Adapter 家族。 |
| `rank` | `int` (1-512) | `32` | Adapter rank / dim。 |
| `alpha` | `int (>=1)` | `16` | LoRA scaling alpha(等效 LR ≈ alpha / rank)。dp 侧 LoRA 强制 `alpha == rank`。 |
| `targetUnet` | `bool` | `true` | 训 UNet / transformer。 |
| `targetTextEncoder` | `bool` | `false` | 训 text encoder。dp / Anima 忽略此字段 — TE 始终冻结。 |
| `networkDropout` | `float (0-1)` | `0.0` | adapter dropout 做正则。 |
| `convDim` / `convAlpha` | `int \| None` | `None` | 仅 `locon` / `loha` 有效。 |

## `optimizer`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `str` | `"adamw8bit"` | 后端能识别的任意 optimizer 名。dp 接受 `adamw` / `adamw8bit` / `adamw_optimi` / `stableadamw` / `lion` / `prodigy` 等。 |
| `lr.unet` | `float` | `1e-4` | UNet / transformer 学习率。 |
| `lr.textEncoder` | `float` | `5e-5` | text encoder 学习率。 |
| `schedule` | `str` | `"cosine_with_restarts"` | LR scheduler 名。 |
| `warmupSteps` | `int` | `100` | 进 schedule 前的线性 warmup。 |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Adam betas。 |
| `weightDecay` | `float` | `0.0` | L2 weight decay。 |
| `eps` | `float` | `1e-8` | optimizer eps。 |
| `optimizerArgs` | `dict[str, str]` | `{}` | 透传给 optimizer 的自由 `key=value` 包。 |
| `gradientRelease` | `bool` | `false` | dp `gradient_release`:分块释放梯度,省显存。 |

## `schedule`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `epochs` | `int (>=1)` | `10` | 训练 epoch 数。 |
| `batchSize` | `int (>=1)` | `1` | per-step batch size。 |
| `gradAccum` | `int (>=1)` | `2` | 梯度累积步数。 |
| `maxSteps` | `int \| None` | `None` | 可选全局 step 上限。 |

## `sampling`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `enabled` | `bool` | `true` | 训练中生成 preview。 |
| `everyNEpochs` | `int (>=1)` | `1` | 按 epoch 取样节奏。 |
| `everyNSteps` | `int \| None` | `None` | 按 step 取样节奏。live preview 下与 `everyNEpochs` 互斥。 |
| `promptsFile` | `Path \| None` | `None` | 纯文本 prompt 文件(每行一条,kohya 格式)。 |
| `resolution` | `list[int]` | `[1024, 1024]` | 取样分辨率。 |
| `seed` | `int` | `42` | 取样 RNG 种子。 |
| `enableLiveInference` | `bool` | `false` | 开 lorahub 侧 preview worker(仅在 `backend.type == "diffusion-pipe"` 时有意义)。 |
| `inferenceSteps` | `int (>=1)` | `24` | 当 prompt 行没写 `--s` 时的默认采样步数。 |
| `inferenceCfg` | `float (>0)` | `5.0` | 当 prompt 行没写 `--l` 时的默认 CFG。 |

## `output`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `name` | `str` | `"lora_output"` | LoRA 文件名 stem。 |
| `saveEveryNEpochs` | `int (>=1)` | `1` | epoch 节奏 checkpoint。 |
| `saveEveryNSteps` | `int \| None` | `None` | step 节奏 checkpoint。**优先于 epoch 节奏**:设了它之后,编译器会丢掉 epoch flag,避免对齐边界双写。 |
| `saveDtype` | `Literal["fp16", "bf16", "float"]` | `fp16` | 落盘权重 dtype。 |
| `outputDir` | `Path \| None` | `None` | 覆盖 workspace 输出目录。 |

## `backend`

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `type` | `Literal["kohya", "diffusion-pipe", "anima_lora"]` | `kohya` | 训练引擎。 |
| `pinVersion` | `str \| None` | `None` | 把后端钉到某次 commit。 |
| `sdScriptsPath` | `Path \| None` | `None` | 覆盖 kohya checkout 路径。 |
| `pythonExecutable` | `Path \| None` | `None` | 覆盖 kohya / dp / anima_lora 的 venv Python。 |
| `extraArgs` | `dict[str, Any]` | `{}` | 后端 CLI flag 的逃生口。 |
| `diffusionPipe` | `DiffusionPipeOptions \| None` | `None` | dp 专属(见下)。kohya 忽略。 |
| `animaLora` | `AnimaLoraOptions \| None` | `None` | anima_lora 专属（见下）。kohya / dp 忽略。 |

## `backend.diffusionPipe`

仅在 `backend.type == "diffusion-pipe"` 时消费。要点字段:

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `pipelineStages` | `int (>=1)` | `1` | DeepSpeed pipeline 并行度。 |
| `gradientClipping` | `float (>0)` | `1.0` | grad-norm 截断值。 |
| `cachingBatchSize` | `int (>=1)` | `1` | 缓存 latent / TE embedding 时的 batch。 |
| `stepsPerPrint` | `int (>=1)` | `1` | DeepSpeed 日志节奏。 |
| `blocksToSwap` | `int (>=0)` | `0` | offload 到 RAM 的块数(0 = 关)。 |
| `compile` | `bool` | `false` | **保持关闭**。dp pipeline 内部已经 compile;打开会再起一个 Inductor 编译池,在 Anima 跑的 cache 尾段卡在 `unix_stream_data_wait`。 |
| `minAr` | `float (>0)` | `0.5` | 最小 aspect ratio bucket。 |
| `maxAr` | `float (>0)` | `2.0` | 最大 aspect ratio bucket。 |
| `numArBuckets` | `int (>=1)` | `7` | aspect-ratio 桶数。**latent cache 占盘的最大变量** — 每个桶都存自己一份 VAE 编码;200 张图 × 11 桶在训练前就要 ~35 GB cache。Anima recipe 选 `5`。 |
| `cacheShuffleNum` | `int (>=0)` | `0` | 缓存阶段随机打乱前 N 个 tag(0 保持原顺序)。 |
| `skipEmptyCaption` | `bool` | `true` | 跳过没 caption 的图;`false` 时按空 caption 训。 |
| `checkpointEveryNMinutes` | `int (>=1) \| None` | `None` | **完整 DeepSpeed checkpoint** 的 wall-clock 节奏(optimizer state + LR scheduler + dataloader epoch + `latest` 指针)。`POST /api/jobs/{id}/resume` 必需 — `output.saveEveryNSteps` 只写 LoRA 权重,不写 optimizer 状态。Anima recipe 选 `30`。 |
| `checkpointEveryNEpochs` | `int (>=1) \| None` | `None` | 同上,但按 epoch 节奏。两者只用其一(单 epoch 长跑选 minutes 更稳)。 |
| `multiNode` | `MultiNodeOptions \| None` | `None` | 多机训练(B8):`{hostfile, num_nodes, master_addr?, master_port?}`。runner 透传给 DeepSpeed launcher。 |
| `modelPaths` | `dict[str, str]` | `{}` | 自由 per-arch 路径包。**key 原样写入 dp TOML**,沿用上游 `snake_case`(`transformer_path` / `vae_path` / `llm_path`)。 |

`wandb_api_key` 故意不在 schema 中;diffusion-pipe 直接读 `$WANDB_API_KEY`,
密钥不进磁盘 TOML。

## `backend.animaLora`

仅在 `backend.type == "anima_lora"` 时消费。完整字段表（method / preset 枚举、network / optim / 缓存 / 注意力 / offload / 验证段，以及五个 method 的子配置 discriminated union）由 [`AnimaLoraOptions`](https://github.com/GALIAIS/LoraHub/blob/main/lorahub/core/config/schema.py) 定义。

简而言之:

- `method: Literal["lora", "postfix", "chimera", "easycontrol", "ip_adapter"]`
- `preset: Literal["default", "low_vram", "graft", "half", "quarter", "tenth", "debug"]`
- 14 个 lock 字段在 UI 上以锁标(🔒 / ⚠️)呈现,锁住后由 LoraHub 强制成
  上游推荐值,避免误改。

## `resume`

`resume` 控制 resume 用的状态写入。`saveState == true` 时,kohya 把 optimizer
+ scheduler 状态写在 safetensors 旁边,后续运行可以从中断点继续。状态目录较
大;磁盘紧张就用 `saveStateEveryNEpochs` 限频。

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `saveState` | `bool` | `true` | 每个 checkpoint 都写 optimizer / scheduler 状态。 |
| `saveStateAtEnd` | `bool` | `true` | 训练正常结束时强制写一次状态。 |
| `saveStateEveryNEpochs` | `int \| None` | `None` | 把状态写入限到每 N 个 epoch。 |
| `resumeFrom` | `Path \| None` | `None` | 本地 resume 路径。 |
