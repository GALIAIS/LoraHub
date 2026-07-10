---
title: 安装
description: 安装 LoraHub 并准备至少一个训练后端。
---

# 安装

LoraHub 需要 Python 3.11 或 3.12，配合一块显存 ≥ 8 GB 的 NVIDIA GPU。Windows、Linux、macOS 三个平台都支持。至少要装一个训练后端：

- `kohya-ss/sd-scripts` — 覆盖 SD / SDXL / Flux / Lumina / HunyuanImage / Anima。
- `tdrussell/diffusion-pipe` — 覆盖现代 DiT 阵列（Flux2 / Chroma / Wan / Cosmos / Anima 等）。
- `sorryhyun/anima_lora` — Anima 专用栈，已 vendored 在 `external/anima_lora/`，无需 clone。

三个后端可共存，每份配置通过 `backend.type` 字段切换。

## 一键脚本（推荐）

仓库根目录的 `scripts/install.{bat,sh}` 会从零搭好运行环境：通过 uv 安装 Python 3.12、创建 `.venv`、装 Python 依赖、装便携 Node.js、装前端依赖。所有产物都落在项目根目录的 `.lorahub/`、`.venv/`、`.node/` 下，整个项目目录可以打包搬到另一台同架构的机器直接运行。

=== "Windows"

    ```powershell
    git clone https://github.com/GALIAIS/LoraHub
    cd LoraHub
    scripts\install.bat
    ```

=== "Linux / macOS"

    ```bash
    git clone https://github.com/GALIAIS/LoraHub
    cd LoraHub
    bash scripts/install.sh
    ```

装好后用 `scripts\run.bat` / `scripts/run.sh` 启动：

```text
scripts\run.bat              # 默认生产模式：API + 已构建的 SPA
scripts\run.bat dev          # 开发模式：API + Vite 热更新
scripts\run.bat api          # 仅启动 API
```

## 手动安装

如果机器已有 Python 3.11/3.12 与 Node.js 20+，可以直接走 pip：

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[api,dev,cpu]"
cd web && npm install && cd ..
```

## 后端 bootstrap

=== "在工作树内 bootstrap"

    ```powershell
    lorahub bootstrap-kohya              # 约 10 分钟：clone + venv + PyTorch + 依赖
    lorahub bootstrap-diffusion-pipe
    ```

    `bootstrap-kohya` 默认 PyTorch 2.6.0 + CUDA 12.4。可用 `--cuda cu121` / `cu118` / `cu128`、`--torch X.Y.Z` 切版本，`--no-xformers` 跳过可选 xformers，`--force` 抹掉半装的目录重来。

=== "指向已有 checkout"

    ```powershell
    set LORAHUB_KOHYA_SD_SCRIPTS=C:\path\to\sd-scripts
    set LORAHUB_DIFFUSION_PIPE=C:\path\to\diffusion-pipe
    ```

    Linux/macOS 用 `export`。也可把 `.env.example` 复制为 `.env` 后编辑——LoraHub 启动时会从项目根读取。

### `anima_lora` 的特殊之处

`external/anima_lora/` 已随 LoraHub 一起分发。它需要独立的 venv（CPython 3.13 + torch 2.11/2.12 nightly + CUDA 13.x），与 LoraHub 主 venv 隔离。在 Web UI 的「设置 → 安装」面板点击「安装 anima_lora」会自动跑 `uv sync` 在 `external/anima_lora/.venv` 内创建这个 venv，首次安装大约下载 6–8 GB（torch + CUDA wheels）。

venv 装好后，安装面板会出现「下载模型」按钮，从 HuggingFace `circlestone-labs/Anima` 仓库拉取约 14 GB 的 Anima 基础模型（DiT、Qwen3 文本编码器、Qwen-Image VAE）到项目根目录的 `models/` 下。下载完成后即可启动 anima 训练。

如需指向已有的 anima venv：

```powershell
set LORAHUB_ANIMA_LORA_PYTHON=C:\path\to\anima_lora\.venv\Scripts\python.exe
```

LoraHub 自身不会 import anima_lora 的代码，只把它作为子进程拉起。

## 可选 extras

| Extra     | 何时安装                                       | 命令                            |
| --------- | ---------------------------------------------- | ------------------------------- |
| `api`     | FastAPI 服务（`lorahub serve`）                | `pip install -e ".[api]"`       |
| `cpu`     | WD14 标注通过 CPU 版 `onnxruntime`             | `pip install -e ".[cpu]"`       |
| `gpu`     | WD14 标注通过 `onnxruntime-gpu` 走 CUDA        | `pip install -e ".[gpu]"`       |
| `tagging` | JoyTag（PyTorch）标注后端                      | `pip install -e ".[tagging]"`   |
| `dev`     | 测试、lint、mypy、httpx                        | `pip install -e ".[dev]"`       |
| `docs`    | 构建本站文档                                   | `pip install -e ".[docs]"`      |

`cpu` 与 `gpu` 提供同名模块，不可共存。切换到 GPU 版前先执行
`pip uninstall onnxruntime`，再安装 `.[gpu]`。

## 下一步

- [快速开始](quickstart.md)：四条命令出第一份配置。
- [端到端教程](smoke-test.md)：从原始图片到训练好的 LoRA 全流程。
