---
title: API 端点
description: REST + SSE + WebSocket 端点参考。
---

# API 端点

所有端点挂 `/api`。下面是速查;`lorahub serve` 在线时 `/openapi.json` 是权威。

## Health & settings

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/health` | 服务器状态、版本、后端探测。 |
| GET    | `/api/settings` | 工作台默认值(kohya / dp 路径、Python exe、tagger device、`max_concurrent_jobs`)。 |
| PUT    | `/api/settings` | 写工作台默认值。 |

## Configs

端点集已从 `/api/recipes` 改名 `/api/configs`,与磁盘 `configs/` 对齐。
validator + schema 同时接受 `camelCase`(新 wire format)和 `snake_case`。

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/configs/schema` | 可视化编辑器用的 JSON Schema。 |
| GET    | `/api/configs` | 列 config YAML。 |
| GET    | `/api/configs/{name}` | 看单 config(返回解析后的 camelCase body)。 |
| POST   | `/api/configs/validate` | 校验内存里的 config,返回 Pydantic 字段错误。 |
| POST   | `/api/configs` | 把校验过的 config 存到 `configs/<name>.yaml`。 |
| POST   | `/api/configs/import` | 上传已有配置文件。 |
| POST   | `/api/configs/{name}/duplicate` | 复制一份并改名。 |
| POST   | `/api/configs/{name}/rename` | 重命名。 |
| DELETE | `/api/configs/{name}` | 删配置。 |
| GET    | `/api/configs/templates` | 列内置模板及 placeholder 元数据。 |
| POST   | `/api/configs/templates/{template_id}/instantiate` | 填 placeholder + 落盘。 |

## Jobs

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/jobs` | 列训练任务(运行中 + 历史)。 |
| GET    | `/api/jobs/{id}` | 看单 job。 |
| POST   | `/api/jobs` | 启动 job — body: `{config, workspace?}`。 |
| POST   | `/api/jobs/{id}/rerun` | 原地重跑;清 `events.jsonl`(plain rerun);resume 时不清。 |
| POST   | `/api/jobs/{id}/resume` | 从最近 checkpoint 续训。 |
| POST   | `/api/jobs/{id}/reveal` | 在 host 文件管理器打开 workspace。 |
| GET    | `/api/jobs/{id}/events` | 内存 ring 中的近期事件。 |
| GET    | `/api/jobs/{id}/files` | 列 workspace 内文件。 |
| GET    | `/api/jobs/{id}/files/raw` | 流单个 workspace 文件。 |
| GET    | `/api/jobs/{id}/metrics` | 用于图表的 per-step 聚合指标(loss、val_loss、gpu_samples、samples、checkpoints、overfit_signal)。 |
| GET    | `/api/jobs/{id}/analysis` | 缓存的 AI 训练分析(Markdown)。 |
| POST   | `/api/jobs/{id}/analyze` | 生成或重算 AI 训练分析。 |
| DELETE | `/api/jobs/{id}` | 取消运行中的 job(已结束的传 `?archive=true` 归档)。 |
| POST   | `/api/jobs/{id}/kill` | SIGKILL 进程组;trainer 卡死时的最后手段。 |

## Sweeps

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/sweeps` | 列 sweep 计划(运行中 + 历史)。 |
| GET    | `/api/sweeps/{id}` | 看 sweep 详情(trials + scores)。 |
| POST   | `/api/sweeps` | 提交 sweep:`{config, axes, mode: grid/random/tpe, n_trials?, seed?}`。 |
| GET    | `/api/sweeps/{id}/pareto` | 多目标 sweep 的 Pareto 前沿(B4 cut3)。 |
| DELETE | `/api/sweeps/{id}` | 取消 sweep。 |

## Datasets

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/datasets/scan` | 扫目录,返回图片 + caption 元数据。 |
| GET    | `/api/datasets/thumb` | 流缓存缩略图。 |
| GET    | `/api/datasets/caption` | 读图旁的 `.txt` caption。 |
| PUT    | `/api/datasets/caption` | 写图旁的 `.txt` caption。 |

## Image Studio

数据集管理工作台,带虚拟化网格、多选、拖拽上传与若干 AI 批处理端点。所有
路径都解析在 `LORAHUB_DATASETS_ROOT`(或项目 `./datasets`)下。

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/image-studio/datasets` | 列配置中根下所有数据集。 |
| GET    | `/api/image-studio/list` | 分页列出数据集目录中的图。 |
| GET    | `/api/image-studio/image` | 看单图(分辨率、caption、annotation)。 |
| POST   | `/api/image-studio/annotation` | upsert favorite / soft-delete / quality 字段。 |
| POST   | `/api/image-studio/op/add` | 入队一项操作(rotate / flip / delete / caption / favorite)。 |
| POST   | `/api/image-studio/op/apply` | 对单图应用入队操作。 |
| POST   | `/api/image-studio/batch/delete` | 按路径列表 soft-delete(挪到 `_image_studio_trash/`)。 |
| POST   | `/api/image-studio/upload` | multipart 上传到数据集;自动解压归档。 |
| POST   | `/api/image-studio/ai/caption` | 纯 VLM caption。 |
| POST   | `/api/image-studio/ai/quality` | 经 AI router 的批量画质评分。 |
| POST   | `/api/image-studio/ai/smart-caption` | WD14 + 视觉 LLM 联合 caption(Anima 格式)。模式:`style / character / general`,`triggerWord` 与 `stripStyleTags` 生效。**202 后台 session**:返回 `session_id`,通过 `GET /api/image-studio/ai/smart-caption/status/{id}` 轮询,需要中断走 `POST /api/image-studio/ai/smart-caption/cancel/{id}`。 |
| POST   | `/api/image-studio/ai/smart-caption/single` | 上面的单图变体。 |
| POST   | `/api/image-studio/dedupe/scan` | 异步感知哈希查重。 |
| GET    | `/api/image-studio/dedupe/clusters` | 读查重产出的 cluster。 |
| POST   | `/api/image-studio/similarity/scan` | 异步基于 embedding 的相似度扫描。 |
| GET    | `/api/image-studio/similarity/results` | 读对子相似度结果。 |

## Tagging

| Method | Path | 作用 |
| ------ | ---- | ---- |
| POST   | `/api/tagging/tag` | 启动一次异步 tagging session。默认模型:`SmilingWolf/wd-eva02-large-tagger-v3`。 |
| GET    | `/api/tagging/tag/{session_id}` | 轮询 session 状态。 |

## AI providers + routes

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/ai/providers` / `/api/ai/providers/{id}` | 列 / 看 AI provider。 |
| POST   | `/api/ai/providers` | 建 provider(Anthropic、OpenAI、Vertex、OpenAI-compat 自定义)。 |
| GET    | `/api/ai/providers/{id}/models` | 从 provider 探测可用模型。 |
| POST   | `/api/ai/providers/{id}/test` | 连通性 + 鉴权冒烟。 |
| GET    | `/api/ai/routes` | 列任务路由(`global.default`、`tagging.assist`、`training.analyze` 等)。 |
| PUT    | `/api/ai/routes/{task}` | 把任务绑到 `(provider, model)`。 |
| POST   | `/api/ai/invoke` | 对路由直发同步调用。 |

## Models

| Method | Path | 作用 |
| ------ | ---- | ---- |
| POST   | `/api/models/download` | 启动异步基础模型下载(HF / ModelScope)。 |
| GET    | `/api/models/download/{session_id}` | 轮询下载 session。 |

## Captions

| Method | Path | 作用 |
| ------ | ---- | ---- |
| POST   | `/api/captions/normalize` | 跑离线 caption 变换(shuffle / dropout / booru-alias / keep-tokens)。 |
| POST   | `/api/anima/caption` | Anima 格式 caption 重排。 |

## Samples

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/samples` | 列或流训练期间产出的 sample 图。 |

## System

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/system/stats` | 一次性 host 快照:CPU、RAM、GPU、network、disk。 |
| GET    | `/api/system/attention-backends` | 探当前 GPU 上有哪些 attention kernel(sdpa / xformers / flash / sageattn)可用。 |

## 后端 bootstrap

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/backends` | 支持的训练后端目录(kohya / diffusion-pipe / anima_lora)。 |
| GET    | `/api/backend/bootstrap/status` | 探 kohya / dp 安装是否健康。 |
| POST   | `/api/backend/bootstrap` | 进程内启动 `lorahub bootstrap-*`。 |
| GET    | `/api/runtime/python` | 探 kohya / dp 应使用的 Python 运行时。 |
| POST   | `/api/runtime/python/install` | 装托管 Python 运行时。 |

## Network preset

| Method | Path | 作用 |
| ------ | ---- | ---- |
| GET    | `/api/network/presets` | 内置 network preset(LoRA / LoCon / LoHa / DoRA)。 |
| POST   | `/api/network/probe` | 对底模 checkpoint 探测,建议 network target。 |

## SSE 流(首选)

每条事件带单调递增的 `id: <seq>`。浏览器 `EventSource` 在重连时上送
`Last-Event-ID`,服务器从该索引续传,实现无 gap。

| Path | 载荷 |
| ---- | ---- |
| `/api/jobs/{id}/sse` | 实时 `TrainingEvent` 流;先回放 per-job ring,再转发新事件。包含 `step` / `epoch_end` / `validation` / `checkpoint_saved` / `sample_ready` / `cache_progress` / `oom` / `gpu_sample` / `done` / `preview_unavailable`(B5)等。 |
| `/api/backend/bootstrap/sse` | 当前 `bootstrap-*` session 的进度。 |
| `/api/system/sse` | host 快照,1 Hz;空闲每 25 s 发 `: ping` SSE 注释。 |

## 旧 WebSocket 流(fallback)

| Path | 载荷 |
| ---- | ---- |
| `/api/jobs/{id}/stream` | 与 SSE 同载荷;每 25 s 发 `{type:"ping"}` 心跳。 |
| `/api/backend/bootstrap/stream` | 与 SSE 同载荷。 |
| `/api/system/stream` | 与 SSE 同载荷;pre-SSE 客户端兜底。 |
