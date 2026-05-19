---
title: 配置
description: TrainingConfig schema 是一次 LoRA 训练的单一语义描述。
---

# 配置

一份 *config* 是一个 YAML 文件,描述一次 LoRA 训练的全部输入。LoraHub 用
`TrainingConfig` Pydantic 模型校验该文件,然后由后端专属编译器把它翻成训练
引擎的原生参数(目前是 kohya-ss/sd-scripts、tdrussell/diffusion-pipe,以及
vendored 的 anima_lora)。

schema 的定位是 **语义层**:用户描述要训什么,而不是怎么调后端。新后端接进
来不需要改既有 config 文件。

!!! note "Recipe → config 改名"
    早期文档与代码把这些文件叫"recipe"、放在 `recipes/`。磁盘目录现在叫
    `configs/`,REST 端点挂 `/api/configs`。Python 类型仍叫 `TrainingConfig`
    (一直是)。`recipe` 这个词只作为抽象名词保留在注释里。

## 顶层结构

```yaml
schemaVersion: "1.0"

baseModel:        # 哪份 checkpoint, 什么架构
dataset:          # 图在哪, 分辨率, bucket, caption
network:          # LoRA / LoCon / LoHa / DoRA 形状(rank, alpha, target)
optimizer:        # 类型, 学习率, schedule, warmup
schedule:         # epochs, batchSize, gradAccum
precision:        # fp16 / bf16 / fp32
sampling:         # 训练中的 preview(可选)
output:           # 文件名, 保存节奏, dtype
backend:          # 训练引擎 + extraArgs 逃生口
resume:           # resume 用的 optimizer / scheduler 状态
```

每段都有 SDXL × 8 GB 显存的默认值,所以最小 config 只需要给
`baseModel.checkpoint` 与 `dataset.source`。

## camelCase / snake_case

Pydantic schema 用 `to_camel` alias generator + `populate_by_name=True`,
validator 同时接受两种写法。新配置 emit `camelCase`(对齐前端表单字段与
camelCase API wire);旧 `snake_case` 配置原样 round-trip。

```yaml
# 两种写法都校验为同一个 TrainingConfig
schemaVersion: "1.0"   # 或 schema_version
baseModel:             # 或 base_model
  arch: sdxl
  checkpoint: ./model.safetensors
schedule:
  batchSize: 2         # 或 batch_size
  gradAccum: 2         # 或 grad_accum
```

两个值始终保留字面写法:

- `caption.strategy: tag_file` — Literal,不是字段名。
- `backend.diffusionPipe.modelPaths.transformer_path` / `vae_path` /
  `llm_path` — key 直接写到 dp TOML,上游用 snake_case。

## 一份 config 怎么变成训练运行

1. **Load** — `load_config(path)` 解析 YAML、套默认、校验类型,返回
   `TrainingConfig`。
2. **路径规范化** — job 启动时把 recipe 内每条相对路径(checkpoint、vae、
   `archPaths.*`、`dataset.source`、`modelPaths.*`、`init_from`、
   `prompts_file`)对项目根做绝对化,使训练子进程不论 chdir 到哪儿都能找到。
3. **Compile** — `compile_config(cfg, workspace)` 返回入口 argv 与一组要写到
   workspace 的文件(`dataset.toml`、diffusion-pipe TOML、sample prompts 等)。
4. **Launch** — 选中的后端起子进程,把 stdout 解析成 `TrainingEvent`,持久
   化到 `events.jsonl`。SSE / WS 流在重连时回放该文件。

## 进一步阅读

- [模板](templates.md) — 内置起步模板与 placeholder 体系。
- [字段参考](fields.md) — schema 的每一个旋钮,按段分组。
