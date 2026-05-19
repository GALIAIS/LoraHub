---
title: anima_lora 后端接入
description: 把 anima_lora 作为第三平行训练后端接入 LoraHub 的设计与切分。
---

# anima_lora 后端接入

`sorryhyun/anima_lora` 是 Anima DiT 专用的训练栈,基于 sd-scripts fork
但深度自研,带 OrthoLoRA / T-LoRA / Hydra / Postfix / EasyControl /
IP-Adapter / DMD turbo 等算法,以及 torch.compile full + flash-attn +
unsloth offload + 自定义 down autograd 的工程化加速。

LoraHub 当前的 kohya / dp 后端跑 anima 是能训的,但只覆盖通用 LoRA
路径,享受不到上面那些算法和加速。本文档定义把 anima_lora 作为**第三
平行后端**接入的设计。

## 决策

| 决策项 | 取值 |
| ------ | ---- |
| 接入形态 | 第三 backend(`type: anima_lora`),与 kohya/dp 完全并列 |
| schema 隔离 | `AnimaLoraOptions` 独立成型,不动 kohya/dp 现有字段 |
| 仓库引用 | **vendored**:源码住 `external/anima_lora/`,LoraHub 自带,不再让用户去 clone。LoraHub 项目自身的 fork 副本,可针对 lock/patch 单独改造 |
| Python 解释器 | `LORAHUB_ANIMA_LORA_PYTHON` env 或 `BackendConfig.python_executable` 字段。anima_lora 要求 torch 2.11/2.12 nightly + CUDA 13.x,用户自己起一个 venv 装依赖,LoraHub 主 venv 隔离 |
| 执行模式 | 完全子进程,LoraHub 不 import anima_lora 任何代码 |
| anima arch 兼容 | 保留 kohya/dp 跑 anima 能力,UI/scaffold 默认切到 anima_lora |
| method 表达 | `method: Literal["lora","postfix","chimera","easycontrol","ip_adapter"]`,默认 lora 自动堆叠 OrthoLoRA + T-LoRA |
| preset 表达 | `preset: Literal["default","low_vram","graft","half","quarter","tenth","debug"]`(对齐 anima_lora `presets.toml`) |

## 目录布局(照搬 kohya/dp 模式)

```
lorahub/core/backends/anima_lora/
    __init__.py      # 导出 AnimaLoraBackend
    backend.py       # 实现 TrainingBackend 协议:validate / estimate_vram / launch
    bootstrap.py     # 探测 LORAHUB_ANIMA_LORA_REPO + 校验 train.py / inference.py 存在
    compiler.py      # AnimaLoraOptions → CLI argv + emit anima_lora.toml
    runner.py        # subprocess + stdout 解析为 TrainingEvent
```

新文件,**不动**:

- `lorahub/core/backends/kohya/`(全保留)
- `lorahub/core/backends/diffusion_pipe/`(全保留)
- `lorahub/core/config/schema.py` 既有字段(`BaseModelConfig`、`OptimizationConfig`、`NetworkConfig` 一行不动)

只动一处:

- `BackendConfig.type` Literal 加 `"anima_lora"`,新增 `anima_lora: AnimaLoraOptions | None` 字段(挂法与 `diffusion_pipe: DiffusionPipeOptions | None` 对称)
- `lorahub/api/jobs_helpers.py` 两处 dispatch(`backend_type == "anima_lora"`)

## AnimaLoraOptions 字段全集

按 anima_lora `base.toml` + `lora.toml` + `presets.toml` 抽出。
分组对应上游 TOML 段落,emit 时按段落写出。

### 顶层 — 基础

| 字段 | 类型 | 默认 | 说明 |
| ---- | ---- | ---- | ---- |
| `method` | Literal | `"lora"` | 训练方法选择 |
| `preset` | Literal | `"default"` | 内存/采样比 preset |
| `pretrained_model_name_or_path` | Path | — | DiT 主权重 |
| `qwen3` | Path | — | Qwen3 文本编码器 |
| `vae` | Path | — | QwenImage VAE |
| `t5_tokenizer_path` | Path \| None | None | T5 分词器(可选) |
| `output_dir` | Path | — | 输出目录 |
| `output_name` | str | `"anima_lora"` | 输出文件名 stem |

### Network

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `network_module` | str | `"networks.lora_anima"` |
| `network_dim` | int | 16 |
| `network_alpha` | int | 16 |
| `network_train_unet_only` | bool | true |
| `use_ortho` | bool | true (method=lora 默认) |
| `min_rank` | int | 8 (T-LoRA) |
| `alpha_rank_scale` | float | 1.0 |

### Optim

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `optimizer_type` | Literal | `"AdamW"` |
| `lr_scheduler` | Literal | `"constant"` |
| `learning_rate` | float | 5e-5 |
| `max_train_epochs` | int | 8 |
| `save_every_n_epochs` | int | 2 |
| `checkpointing_epochs` | int | 2 |
| `caption_dropout_rate` | float | 0.1 |

### 采样 / 损失

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `timestep_sampling` | Literal | `"sigmoid"` |
| `sigmoid_scale` | float | 1.0 |
| `discrete_flow_shift` | float | 1.0 |
| `weighting_scheme` | Literal\|None | None |
| `logit_mean` | float\|None | None |
| `logit_std` | float\|None | None |
| `mode_scale` | float\|None | None |
| `vr_loss_weight` | float\|None | None |

### 缓存 / 数据

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `cache_latents` | bool | true |
| `cache_latents_to_disk` | bool | true |
| `cache_text_encoder_outputs` | bool | true |
| `cache_text_encoder_outputs_to_disk` | bool | true |
| `cache_llm_adapter_outputs` | bool | true |
| `use_shuffled_caption_variants` | bool | true |
| `sample_ratio` | float\|None | None (preset 覆盖) |
| `dataset_config` | Path | — |
| `static_token_count` | int | 4096 |
| `vae_chunk_size` | int | 64 |
| `vae_disable_cache` | bool | false |
| `no_half_vae` | bool | false |

### 注意力 / 编译

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `attn_mode` | Literal | `"flash"` |
| `xformers` | bool | false |
| `split_attn` | bool | false |
| `compile_mode` | Literal\|None | None |
| `compile_inductor_mode` | Literal\|None | None |
| `dynamo_backend` | str\|None | None |
| `use_custom_down_autograd` | bool | true |

### 内存 / offload

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `blocks_to_swap` | int | 0 |
| `gradient_checkpointing` | bool | false |
| `unsloth_offload_checkpointing` | bool | false |
| `cpu_offload_checkpointing` | bool | false |
| `mixed_precision` | Literal | `"bf16"` |

### 验证

| 字段 | 类型 | 默认 |
| ---- | ---- | ---- |
| `use_cmmd` | bool | false |
| `validation_seed` | int\|None | None |
| `validation_sample_steps` | int\|None | None |
| `validation_cfg_scale` | float\|None | None |

### Method 子配置 — discriminated union

`method` 的取值决定哪一个 `AnimaLoraMethod*Config` 子结构生效。其它的
不出现在 schema 表面。

#### method = "lora" (默认堆叠 OrthoLoRA + T-LoRA)

无需额外字段,`use_ortho`、`min_rank`、`alpha_rank_scale` 已在 Network 段。

#### method = "postfix"

| 字段 | 默认 |
| ---- | ---- |
| `postfix_mode` (Literal "postfix" \| "cond") | `"cond"` |
| `cond_hidden_dim` | 1024 |
| `splice_position` (Literal) | `"front_of_padding"` |
| `ortho_basis` (Literal) | `"svd_te"` |
| `te_cache_dir` | Path |
| `svd_num_files` | 1024 |
| `lambda_init` | 0.3 |

#### method = "chimera"

| 字段 | 默认 |
| ---- | ---- |
| `use_chimera_hydra` | true |
| `balance_w_content` | 2e-7 |
| `balance_w_freq` | 5e-7 |
| `balance_loss_warmup_ratio` | 0.4 |
| `fei_feature_dim` | 2 |
| `sigma_feature_dim` | 16 |

#### method = "easycontrol"

| 字段 | 默认 |
| ---- | ---- |
| `use_easycontrol` | true |
| `b_cond_init` | -10.0 |
| `cond_scale` | 1.0 |
| `apply_ffn_lora` | 1 |
| `cond_token_count` | 4096 |
| `easycontrol_drop_p` | 0.1 |
| `easycontrol_cond_noise_max` | 0.3 |

#### method = "ip_adapter"

| 字段 | 默认 |
| ---- | ---- |
| `use_ip_adapter` | true |
| `ip_encoder` (Literal) | `"PE-Core-L14-336"` |
| `ip_resampler_layers` | 2 |
| `ip_resampler_heads` | 8 |
| `ip_scale` | 1.0 |
| `ip_image_drop_p` | 0.05 |
| `gate_lr` | 1e-3 |
| `ip_features_cache_to_disk` | bool | true |

DMD turbo 走独立路径(`scripts/distill_turbo.py`),schema 与训练分开。
留 cut4。

## 编译 / spawn 流程

1. 用户提交 TrainingConfig,backend.type == `"anima_lora"`,`backend.anima_lora` 已填。
2. `compiler.py.compile()`:
   - 写 `<workspace>/anima_lora.toml` (合并 base + method + preset 三层成单文件,不模拟 anima_lora 自己的 merge chain — LoraHub 的 schema 已是合并后的视图)。
   - 返回 argv: `[<python>, <repo>/train.py, "--config_file", "<workspace>/anima_lora.toml", ...override...]`
   - `--config_file` 走 anima_lora 已有的 TOML 加载;CLI override 用于动态字段(如 `--seed` 跑 sweep 时)。
3. `runner.py.launch()`:
   - subprocess.Popen,捕获 stdout/stderr,解析 anima_lora 的进度日志格式 → TrainingEvent(`step` / `epoch` / `validation` / `checkpoint`)。
   - LoraHub 的 events.jsonl 写入与 kohya/dp 一致,前端 SSE 端到端无感。

## 探测 (bootstrap)

`AnimaLoraBackend.validate()` 调 bootstrap:

- 仓库路径默认指向 `<lorahub_repo_root>/external/anima_lora/`(vendored)。
  环境变量 `LORAHUB_ANIMA_LORA_REPO` 仅在用户主动覆盖时生效(开发场景:
  外挂另一份 anima_lora,临时调试用)。
- 探测 `<repo>/train.py`、`<repo>/inference.py`、
  `<repo>/library/anima/__init__.py` 是否存在。**不再做"拉取 / uv sync"
  这一步** — vendored 副本随 LoraHub 仓库一起来,不需要二次 fetch。
- 解析 Python 解释器路径(`LORAHUB_ANIMA_LORA_PYTHON` env 或
  `BackendConfig.python_executable`)。**没有自动安装依赖逻辑** — 用户
  应该自己针对那个 venv 跑过 `uv sync`。
- 检查 `<python>` 可执行,但**不**强制 `import library.anima` 成功
  (kohya bootstrap 是这样做的);因为 LoraHub 主进程 venv 一般装不了
  anima_lora 的 nightly torch,这种 import-side check 在主 venv 里必失败。
  改成"语法 sanity":`<python> -c "import sys; print(sys.version)"`,
  以及"vendored 副本完整性":必要文件存在 + Python 版本不低于 3.13。
- 失败返回 ValidationIssue(severity=error)。

## VRAM 估算

照搬 `_common/vram.py` 现有 anima 行 (`2000 model MiB / 768 per-block`)。
unsloth offload 开启时把 activations 估值打 0.6 折,粗略对齐上游 13.4 GB
峰值。

## Preview backend

cut3 实现 `AnimaLoraInferenceBackend`,注册到 B5 表,优先级高于现有
`AnimaInferenceBackend`(后者是 LoraHub 内嵌的简化版)。子进程调
`<repo>/inference.py`,可选 Spectrum 推理。

## Cut 切分

| Cut | 内容 | 测试 | 时长 |
| --- | ---- | ---- | ---- |
| **cut0** | 本文档 + AnimaLoraOptions 全字段 schema + BackendConfig.type 枚举值 + dispatch NotImplementedError 占位 | schema 单元 + dispatch raise 单元 | 0.5d ✅ |
| **cut1** | compiler + emit anima_lora.toml + method 路由 | 编译器逐 method snapshot | 1d ✅ |
| **cut2** | bootstrap + backend.py + runner + jobs_helpers dispatch + VRAM | spawn 单元(patch subprocess)+ 集成测试(最短 step) | 1.5d ✅ |
| **cut3** | preview backend 注册到 B5 表 | 注册表优先级单元 + spawn smoke | 0.5d ✅ |
| **cut4** | DMD turbo (`scripts/distill_turbo.py`) 独立路径 | schema + compiler 分支 + turbo runner + parser + 测试 | 1d ✅ |

## 已知风险

- **torch 版本**:anima_lora 要求 torch 2.11/2.12 nightly + CUDA 13.x。LoraHub 主 venv 不一定达标 — 走子进程 + 用户自己的 anima_lora venv 规避;LoraHub 不 import anima_lora 代码。
- **数据预处理**:LoraHub 的 anima_lora 后端在 `launch()` 进入主训练前会**自动**检测 `<workspace>/post_image_dataset/lora/{stem}_anima_te.safetensors` 是否齐全,缺则依次 spawn 上游 `preprocess/resize_images.py` → `cache_latents.py` → `cache_text_embeddings.py`,把 cache 写到 `<workspace>/post_image_dataset/lora/`。compiler 同时通过 `--source_image_dir` / `--resized_image_dir` / `--lora_cache_dir` 三个 CLI 覆盖把上游 `base.toml` 的相对路径锁到这套 LoraHub 管理的绝对路径。这样 `cfg.dataset.source` 与 kohya / dp 完全一致(始终指向**原始图片目录**),用户切后端不必重写 recipe 也不必手跑 `make preprocess`。preprocess 失败 → `PreprocessError` → backend.launch 返回 `CompilationError`,训练不会启动。
- **stdout 格式**:anima_lora 的训练日志格式可能与 kohya 不同。runner.py 的 parser 要在 cut2 抓真实日志锁定;留 fallback 模式(parse 失败时仍写原始行到 events)。
- **schema 膨胀风险**:method 子配置走 discriminated union,Pydantic 2 原生支持。不会影响 kohya/dp 路径的字段数。
- **模型路径**:三个 yaml 模板默认走 `./models/circlestone-labs__Anima/split_files/...`,与 kohya / dp 模板共用 LoraHub 项目根 `./models/`,单一真相源。不再依赖 `external/anima_lora/models/` 子目录。
