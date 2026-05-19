# LoraHub

自托管的 LoRA 训练工作台，把 kohya / diffusion-pipe / anima_lora 三个训练后端套在同一份配置 schema、同一套 REST + SSE API 和同一个 React Web UI 之下。

A self-hosted LoRA training workbench that wraps three training backends (kohya, diffusion-pipe, anima_lora) behind one config schema, one REST + SSE API, and one React web UI.

[![CI](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml/badge.svg)](https://github.com/GALIAIS/LoraHub/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

---

## 安装 / Install

需要 Python 3.11 或 3.12，Windows / Linux / macOS 可用，训练时需要一块 8 GB+ 显存的 NVIDIA GPU。至少安装一个训练后端。

Requires Python 3.11 or 3.12 on Windows, Linux, or macOS, and an NVIDIA GPU with 8 GB+ VRAM for training. At least one training backend must be available.

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
```

引入后端：

Bring in a backend:

```powershell
lorahub bootstrap-kohya              # SD1.5 / SDXL / SD3 / FLUX / Lumina / HunyuanImage / Anima
lorahub bootstrap-diffusion-pipe     # diffusion-pipe + DeepSpeed pipeline
# anima_lora vendored at external/anima_lora; no separate clone needed
```

或者直接指向已有的 checkout：

Or point at existing checkouts:

```powershell
$env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
$env:LORAHUB_DIFFUSION_PIPE   = "C:\path\to\diffusion-pipe"
```

可选依赖：

Optional extras:

| Extra     | Purpose                                       |
| --------- | --------------------------------------------- |
| `api`     | FastAPI server (`lorahub serve`)              |
| `gpu`     | WD14 tagger on CUDA via `onnxruntime-gpu`     |
| `tagging` | JoyTag (PyTorch) tagger backend               |
| `dev`     | tests, lint, mypy, httpx                      |
| `docs`    | mkdocs documentation site                     |

---

## 快速开始 / Quick start

```powershell
lorahub init my_character
notepad configs/my_character.yaml
lorahub validate configs/my_character.yaml
lorahub train    configs/my_character.yaml
lorahub serve    --port 18765        # optional: web UI + REST API
```

启动后访问 `http://127.0.0.1:18765`。Job 详情页带 SSE 实时事件流，断线后通过 `Last-Event-ID` 续传。

The web UI lives at `http://127.0.0.1:18765`. Job detail pages stream events over SSE and resume on reconnect via `Last-Event-ID`.

---

## 核心特性 / Features

中文：

- 三后端切换：在 设置 → 后端配置 选择 kohya、diffusion-pipe 或 anima_lora，配置 schema 共用。
- 配置编辑器：每个字段带 schema / 后端可见性过滤，锁定字段和警示字段都有徽标提示。
- 图像工作台：WD14（默认 wd-eva02-large-v3）/ JoyTag 标注、smart-caption（WD14 + 视觉 LLM）、phash 去重、批量质量评分。
- 任务调度：单 GPU 队列，断点续训，多 GPU 通过 `CUDA_VISIBLE_DEVICES` 切片，重启后非终止任务标记为 `interrupted`。
- 实时事件：`/api/jobs/{id}/sse`、`/api/system/sse`、`/api/backend/bootstrap/sse`，WebSocket 作为 fallback 保留。
- 超参 sweep：grid、random、Optuna TPE 三种模式，TPE 提供 Pareto 视图，进度跨重启恢复。
- 多机 DeepSpeed launcher（B8）：用 `lorahub jobs ls/cancel/kill/resume/rerun` 与 `lorahub sweeps submit/ls` 在 CLI 操作。
- Anima 完整工作流：模型下载、Anima transformer + Qwen-Image VAE + Qwen3 文本编码器组合、训练时 checkpoint 间 PNG 预览。

English:

- Three backends: pick `kohya`, `diffusion-pipe`, or `anima_lora` per config; the schema layer is shared.
- Visual config editor with schema- and backend-aware field filters and lock / warn badges per field.
- Image Studio: WD14 (default `wd-eva02-large-v3`) and JoyTag taggers, smart-caption (WD14 + vision LLM), phash de-duplication, batch quality scoring.
- Job scheduling: single-slot queue, checkpoint resume, multi-GPU via per-slot `CUDA_VISIBLE_DEVICES`, non-terminal jobs surface as `interrupted` after restart.
- Live events: SSE on `/api/jobs/{id}/sse`, `/api/system/sse`, `/api/backend/bootstrap/sse`; WebSocket kept as fallback.
- Hyperparameter sweeps with grid, random, and Optuna TPE modes; TPE has a Pareto view and resumable state across restarts.
- Multi-node DeepSpeed launcher (B8) and a CLI surface (`lorahub jobs`, `lorahub sweeps`, `lorahub system`).
- Anima end-to-end flow: model download, paired Anima transformer + Qwen-Image VAE + Qwen3 text encoder, PNG preview between checkpoints.

---

## 训练后端 / Training backends

| Backend          | Upstream                                                                | Coverage                                                                  | Notes                                              |
| ---------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| `kohya`          | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)           | SD1.5, SDXL, SD3, FLUX, Lumina, HunyuanImage, Anima                       | Single-process; LoraHub builds a venv and argv.    |
| `diffusion-pipe` | [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) | DiT zoo plus video: Wan, HunyuanVideo, LTX, Cosmos, Flux2, Chroma, others | DeepSpeed pipeline; multi-node launcher available. |
| `anima_lora`     | [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora)         | Anima DiT only                                                            | Vendored at `external/anima_lora/`; see below.     |

每个后端的字段映射在 `lorahub/core/backends/<name>/compiler.py`，schema 字段过滤在 `lorahub/core/config/schema.py`。

The per-backend field mapping lives in `lorahub/core/backends/<name>/compiler.py`; schema-level field filters live in `lorahub/core/config/schema.py`.

---

## 关于 anima_lora vendored / About the vendored anima_lora

`external/anima_lora/` 是 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 的一个固定 snapshot，按照上游 license 在仓内随附。它扩展了 Anima DiT 的训练手段：OrthoLoRA 正交分解、T-LoRA 低秩组合、Hydra 多头适配、postfix 注入、EasyControl 条件控制、IP-Adapter、DMD turbo 蒸馏。

`external/anima_lora/` is a pinned snapshot of [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora), bundled under its upstream license. It adds Anima-DiT-specific training paths: OrthoLoRA, T-LoRA, Hydra, postfix injection, EasyControl, IP-Adapter, and DMD turbo distillation.

不接受针对 `external/` 的直接 PR——bug 修复请去上游。集成层（`lorahub/core/backends/anima_lora/`）的改动则属于 LoraHub 范畴，欢迎 PR。

We do not accept PRs that modify files inside `external/` directly. Send bug fixes upstream. Changes in the integration layer (`lorahub/core/backends/anima_lora/`) are in scope for this repo.

---

## CLI / API

```powershell
lorahub jobs ls                         # list scheduler jobs
lorahub jobs cancel <id>                # graceful stop
lorahub jobs kill <id>                  # SIGKILL
lorahub jobs resume <id>                # continue from last checkpoint
lorahub jobs rerun <id>                 # re-launch with the same config

lorahub sweeps submit configs/sweep.yaml   # grid / random / TPE
lorahub sweeps ls

lorahub system gpu                      # snapshot CUDA devices
lorahub system info                     # python / torch / backend probes
```

REST 端点全部挂在 `/api`：`/api/configs`、`/api/jobs`、`/api/jobs/{id}/sse`、`/api/sweeps`、`/api/image-studio/*`、`/api/system/sse`、`/api/models/download` 等。默认绑定 `127.0.0.1`，没有内置鉴权——请勿直接暴露到公网。

REST endpoints all live under `/api`: `/api/configs`, `/api/jobs`, `/api/jobs/{id}/sse`, `/api/sweeps`, `/api/image-studio/*`, `/api/system/sse`, `/api/models/download`, and so on. The server binds to `127.0.0.1` by default and has no built-in auth; do not expose it directly to the public internet.

---

## License

LoraHub is licensed under **AGPL-3.0-or-later**. See [`LICENSE`](LICENSE) for the full text. If you modify LoraHub and offer the modified version to users over a network, the AGPL requires you to make the modified source code available to those users.

LoraHub 采用 **AGPL-3.0-or-later** 协议；完整文本见 [`LICENSE`](LICENSE)。如果你修改 LoraHub 并通过网络对外提供服务，AGPL 要求把修改后的源码同时开放给这些用户。

`external/anima_lora/` is governed by its own upstream license; see `external/anima_lora/LICENSE`.

---

## Contributing

PR 与 issue 流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。社区行为规范见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

PR and issue workflow: [CONTRIBUTING.md](CONTRIBUTING.md). Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Acknowledgements

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — primary training engine for SD1.5 / SDXL / SD3 / FLUX / Lumina / HunyuanImage / Anima.
- [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) — DeepSpeed pipeline backend covering the modern image and video DiT zoo.
- [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) — Anima DiT training paths, vendored at `external/anima_lora/`.
- [SmilingWolf WD14 / WD-v3 taggers](https://huggingface.co/SmilingWolf) — booru-style tagger ONNX models.
- [fpgaminer/joytag](https://github.com/fpgaminer/joytag) — JoyTag PyTorch tagger.
- [Optuna](https://optuna.org/) — TPE sampler used by the sweep engine.
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [typer](https://typer.tiangolo.com/), [rich](https://rich.readthedocs.io/) — Python service and CLI foundations.
- [React](https://react.dev/), [Vite](https://vitejs.dev/), [TanStack Query](https://tanstack.com/query) — web workbench foundations.




