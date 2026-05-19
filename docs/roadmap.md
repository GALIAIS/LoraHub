---
title: Roadmap
description: LoraHub 发版计划与当前进度。
---

# Roadmap

LoraHub 按可独立交付的小切片发版,每版是一条端到端可用的工作流。

| 版本 | 范围 | 状态 |
| ---- | ---- | ---- |
| v0.1 | CLI tracer bullet:recipe -> kohya -> LoRA file | shipped |
| v0.2 | FastAPI + React UI、配置编辑器、设置、job monitor | shipped |
| v0.3 | 数据集模块:导入、缩略图、caption 编辑器 | shipped |
| v0.4 | 自动 tagger:WD14、JoyTag | shipped |
| v0.5 | Job 队列 + 多 GPU + checkpoint resume | shipped |
| v0.6 | 配置库 + sample 画廊 | shipped |
| v0.7 | SD1.5 + Pony / Illustrious 子风味 | shipped |
| v0.8 | Flux / SD3 / diffusion-pipe 后端(21 个架构) | shipped |
| v0.9 | SSE 事件流、camelCase config schema、Image Studio | shipped |
| v0.10 | Live preview(Anima)、AI 训练分析、GPU 资源趋势 | shipped |
| v1.0 | Sweep v2、多机、公开文档站 | in progress |

## 接下来

v1.0 的当前焦点:

- **Sweep v2** — `SweepPlan` 网格已可用,贝叶斯 + 早停在补。当前已上线
  TPE(`mode/n_trials/seed`,跨重启自动续投,`/api/sweeps/{id}/pareto`
  端点);ASHA 评估为不可接(trial 是完整子进程,无法 step 级 prune)。
- **多机训练** — DeepSpeed ZeRO-3 + pipeline 并行已经在 dp 内;补的部分
  是多机 launcher 与节点健康看板。schema 加
  `cfg.backend.diffusion_pipe.multi_node = {hostfile, num_nodes,
  master_addr?, master_port?}`,新增 `/api/system/cluster`(B8)。
- **公开文档站** — 本 mkdocs-material 文档随仓库走;v1.0 发到 GitHub
  Pages,做版本快照。

v0.9 / v0.10 已落地的:

- SSE 事件流,`Last-Event-ID` 续传(WebSocket 保留为兜底)。
- camelCase config schema(snake_case 仍由 validator 接受)。
- Image Studio,带 AI smart-caption(WD14 + VLM)、去重、相似度扫描;
  smart-caption 改为后台 session(202 + status + cancel)。
- diffusion-pipe Anima live preview:事件驱动 checkpoint 监视、PEFT-to-
  kohya LoRA 转换、GPU 预算控制。
- AI 训练分析端点,带 loss 曲线解读。
- run-summary 卡片,带 GPU 资源趋势与 overfit 信号。
- B5 PreviewWorker 抽象化:`InferenceBackend` 注册表 +
  `DiffusersInferenceBackend`(SDXL/SD1.5/SD2/SD3/Flux 通用) +
  `preview_unavailable` 事件。视频架构(Wan / HunyuanVideo / LTX /
  Cosmos)在白名单外,显式回退 StubInference。
- B7 image_studio.py 拆成 9 个子模块,27 条路由 URL 不变。
- B9 CLI 与 API 对齐:`lorahub jobs ls/show/cancel/kill/resume/rerun`、
  `lorahub sweeps submit/ls`、`lorahub system gpu/info`。

## 已搁置 / 不做

- DiffSynth-Studio 后端接入:架构覆盖与 dp 重叠高,接入成本超回报。
- 模型文件命名模板(B6):用户暂决定不做。
- ASHA sweep:不可接,见 audit-2026-05。

## 跟踪工作

- 日常变更见
  [`CHANGELOG.md`](https://github.com/GALIAIS/LoraHub/blob/main/CHANGELOG.md)。
- 大块工作走 GitHub Issue:
  [github.com/GALIAIS/LoraHub/issues](https://github.com/GALIAIS/LoraHub/issues)。
