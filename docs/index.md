---
title: LoraHub
description: 面向扩散模型的开源 LoRA 训练工作台。Open-source LoRA training workbench for diffusion models.
hide:
  - navigation
---

# LoraHub

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

面向扩散模型的开源 LoRA 训练工作台:数据集、打标、训练、实时预览、分析,一条流水线。

LoraHub 在三个上游训练栈之上(`kohya-ss/sd-scripts`、`tdrussell/diffusion-pipe`、
`sorryhyun/anima_lora`)叠加一层语义化的配置 schema,以及统一的 CLI / API / Web UI。
目标是让 LoRA 训练可复现、可配方化、不绑定具体后端。anima_lora 已 vendored 在
`external/anima_lora/` 目录,无需用户额外 clone。

```
+---------------------------------------------------+
|  CLI  /  React web UI                             |
+----------------------+----------------------------+
                       | TrainingConfig + SSE 事件
+----------------------v----------------------------+
|  Core: schema · backends · events · inference     |
+----------------------+----------------------------+
                       | subprocess + JSON
+----------------------v----------------------------+
|  Kohya · DiffusionPipe · AnimaLora                |
+---------------------------------------------------+
```

## 状态

**Alpha — 已用于真实 LoRA 工作。**

当前可用的能力:

- React + FastAPI 工作台:Dashboard、Jobs、Configs、Datasets、Image Studio、
  Sample Gallery、Sweeps、Settings(zh-CN UI)。事件流走 SSE,浏览器原生重连
  + `Last-Event-ID` 续传;旧 WebSocket 端点保留为兜底。
- 三个并列后端,共用同一份 config schema:
    - **kohya** — 8 个架构(SD1.5、SDXL、SD3、Flux、Lumina、HunyuanImage、
      Anima),`archVariant` 区分 SDXL 子风味(Pony / Illustrious / NoobAI /
      Animagine)。
    - **diffusion-pipe** — 21 个架构,含 Flux2、Chroma、HiDream、OmniGen2、
      AuraFlow、Qwen-Image、Cosmos / Cosmos Predict2、Wan、LTX / LTX2、
      HunyuanVideo / 1.5、HunyuanImage、Z-Image、ErnieImage,以及上游通过
      cosmos_predict2 通道路由的 Anima。
    - **anima_lora** — Anima 专用栈(vendored 在 `external/anima_lora/`),
      method 枚举 `lora / postfix / chimera / easycontrol / ip_adapter`,
      preset 枚举 `default / low_vram / graft / half / quarter / tenth /
      debug`。
- Anima 全套预览:模型下载器、transformer + Qwen-Image VAE + Qwen3-0.6B 文
  本编码器配套配置、训练、checkpoint 间实时渲染。
- Image Studio:虚拟化网格、多选、拖拽上传、AR-bucket 标注策略
  (style / character / general)、内联 VLM 智能标注(WD14 EVA02 + 视觉 LLM)、
  感知哈希去重、批量画质评分、回收站 + 还原。smart-caption 批处理已迁为后台
  session(`POST 202` + `GET status` + `POST cancel`)。
- 可视化配置编辑器覆盖全部进阶字段;YAML 在 wire 上用 camelCase,validator
  仍接受 snake_case。
- Job 运行时:每槽位 `CUDA_VISIBLE_DEVICES`、checkpoint resume、SSE 事件回
  放、GPU 采样线程、AI 训练分析(Claude 读 metrics + config,返回 Markdown
  诊断)、job 详情页可折叠的 run-summary 卡片。
- diffusion-pipe 训练时 live preview:lorahub 监视 `output/step{N}/` 目录,
  通过子进程调 sd-scripts 的 Anima inference,每个新 checkpoint 按 prompt
  渲一张 PNG;sd-scripts 不可用时回退到 Stub 占位。
- 双后端一键 bootstrap、uv 装依赖、便携 CPython runtime、HF / ModelScope
  下载器、PyPI 镜像探测。
- 778 个测试覆盖 schema、compiler、parser、runner、API router、scheduler、
  sweep、tagger、caption、inference preview、CLI。

!!! warning "尚未提供"
    - 网格之外的随机 / 贝叶斯 sweep 策略(TPE 已上线,ASHA 评估为不可接)。
    - 嵌入式 Weights & Biases 看板。
    - 跑端到端真实训练的 CI(目前只有单元 / 集成测试)。
    - API 的可选认证 / 多用户模式。

## 快速链接

- [安装](getting-started/install.md) — 系统要求、`pip install`、后端
  bootstrap。
- [快速开始](getting-started/quickstart.md) — 四条命令把第一份配置跑成
  第一次训练。
- [冒烟测试](getting-started/smoke-test.md) — 从图片到 LoRA 文件的全
  流程。
- [配置概览](recipes/index.md) — `TrainingConfig` 结构与字段索引。
- [CLI 参考](cli/index.md) — 每条 `lorahub` 命令一例。
- [API 参考](api/index.md) — REST + SSE + WebSocket 端点。
- [Roadmap](roadmap.md) — 下一版的工作。

## 许可证

AGPL-3.0-or-later。详见 [LICENSE](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE)。

---

## English

LoraHub is an open-source LoRA training workbench for diffusion models —
datasets, captioning, training, live previews, and analysis in one
workflow. It wraps three production backends — `kohya-ss/sd-scripts`,
`tdrussell/diffusion-pipe`, and `sorryhyun/anima_lora` (vendored at
`external/anima_lora/`) — behind a stable, semantic config layer and a
unified CLI / API / web UI. The goal is reproducible, recipe-driven LoRA
training that does not lock you into any single backend.

### Status

Alpha, but already in daily use. The React + FastAPI workbench covers
the full loop (dashboard, jobs, configs, datasets, image studio, sample
gallery, sweeps, settings). Three backends share the same config schema
and dispatch through `backend.type`. SSE event streams resume gap-free
via `Last-Event-ID`; legacy WebSocket endpoints stay as a fallback.

What is not yet here: random / Bayesian sweep strategies beyond TPE
(ASHA was scoped and rejected — trials are full subprocesses without
step-level reporting), an embedded W&B dashboard, end-to-end CI that
trains a real LoRA, and optional auth for the API.

### Quick links

- [Install](getting-started/install.md) — requirements, `pip install`,
  backend bootstrap.
- [Quick start](getting-started/quickstart.md) — first config to first
  training run in four commands.
- [Smoke test](getting-started/smoke-test.md) — full path from images
  to a trained LoRA.
- [CLI reference](cli/index.md) — every `lorahub` command with one
  example.
- [API reference](api/index.md) — REST + SSE + WebSocket endpoints.

### License

AGPL-3.0-or-later. See [LICENSE](https://github.com/GALIAIS/LoraHub/blob/main/LICENSE).
