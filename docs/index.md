---
title: LoraHub
description: 自托管的 LoRA 训练工作台。
hide:
  - navigation
---

# LoraHub

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE)

LoraHub 是一个自托管的 LoRA 训练工作台。它把三套训练栈
（[`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts)、
[`tdrussell/diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe)、
[`sorryhyun/anima_lora`](https://github.com/sorryhyun/anima_lora)）放在同一份语义化配置 schema 之下，统一通过 CLI、REST + SSE API 与 React Web UI 操作。`anima_lora` 已 vendored 在 `external/anima_lora/`，无需用户单独 clone。

## 工作面板覆盖

- **配置**：可视化编辑器，每个字段按 schema 与后端可见性过滤，锁定 / 警示字段有徽标提示。
- **数据集与图像工作台**：导入、缩略图、AR-bucket 标注策略（style / character / general）、WD14（默认 `wd-eva02-large-v3`） 与 JoyTag 标注、WD14 + 视觉 LLM 的智能标注、感知哈希去重、批量画质评分、回收站。
- **训练**：单 GPU 队列、断点续训、多 GPU 通过每槽位 `CUDA_VISIBLE_DEVICES` 切片。重启后非终止任务标记为 `interrupted`，残留训练子进程在新一轮 uvicorn 启动时自动回收。
- **实时事件**：`/api/jobs/{id}/sse`、`/api/system/sse`、`/api/backend/bootstrap/sse`，断线后通过 `Last-Event-ID` 续传。WebSocket 端点保留为兜底。
- **超参 sweep**：grid、random、Optuna TPE 三种模式，TPE 提供 Pareto 视图，进度跨重启恢复。
- **样本图与分析**：训练时按 epoch / step 自动出预览图；详情页提供 loss / 吞吐 / GPU 趋势卡片。

## 三个训练后端

| 后端             | 覆盖范围                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| `kohya`          | SD1.5、SDXL、SD3、FLUX、Lumina、HunyuanImage、Anima。`archVariant` 区分 SDXL 子风味（Pony / Illustrious / NoobAI / Animagine）。 |
| `diffusion-pipe` | 现代 DiT 阵列：Flux2、Chroma、HiDream、OmniGen2、AuraFlow、Qwen-Image、Cosmos / Cosmos Predict2、Wan、LTX / LTX2、HunyuanVideo / HunyuanImage、Z-Image、ErnieImage 等。 |
| `anima_lora`     | Anima DiT 专用栈，方法枚举 `lora / postfix / chimera / easycontrol / ip_adapter`，preset 枚举 `default / low_vram / graft / half / quarter / tenth / debug`。 |

每个后端的字段映射在 `lorahub/core/backends/<name>/compiler.py`，schema 字段过滤在 `lorahub/core/config/schema.py`。

## 快速链接

- [安装](getting-started/install.md) — 系统要求、一键脚本、手动安装、后端 bootstrap。
- [快速开始](getting-started/quickstart.md) — 四条命令把第一份配置跑成第一次训练。
- [端到端教程](getting-started/smoke-test.md) — 从图片到 LoRA 文件的全流程示例。
- [配置概览](recipes/index.md) — `TrainingConfig` 结构与字段索引。
- [CLI 参考](cli/index.md) — 每条 `lorahub` 命令的用法。
- [API 参考](api/index.md) — REST + SSE + WebSocket 端点。

## 许可证

AGPL-3.0-or-later。详见 [LICENSE](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE)。
