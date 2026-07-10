# LoraHub

自托管的 LoRA 训练工作台。统一管理 kohya / diffusion-pipe / anima_lora / ai-toolkit 训练后端，提供配置编辑、任务调度、训练分析、数据集处理、模型下载与产物归档。

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

---

## 安装

需要 Windows 或 Linux，训练时需要 NVIDIA GPU。主程序使用 Python 3.11 / 3.12；vendored 后端会维护自己的独立 venv。

### 一键脚本

`scripts/install.bat`（Windows）或 `scripts/install.sh`（Linux）会从零搭好运行环境：通过 uv 安装 Python 3.12、创建 `.venv`、装 Python 依赖、装便携 Node.js、装前端依赖。所有产物落在项目根目录的 `.lorahub/`、`.venv/`、`.node/` 下，完全自包含。

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
scripts\install.bat              # 默认源
scripts\install-cn.bat           # 国内镜像版
```

```bash
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
bash scripts/install.sh
bash scripts/install-cn.sh       # 国内镜像版
```

装好后用 `scripts\run.bat` / `scripts/run.sh` 启动：

```text
scripts\run.bat              # 默认生产模式：API + 已构建的 SPA
scripts\run.bat dev          # 开发模式：API + Vite 热更新
scripts\run.bat api          # 仅启动 API
```

或注册全局 `lorahub` CLI（推荐）：

```bash
# Linux / macOS / WSL
.venv/bin/python -m lorahub manage install   # → ~/.local/bin/lorahub

# Windows
.venv\Scripts\python.exe -m lorahub manage install   # → %LOCALAPPDATA%\lorahub\bin\lorahub.cmd

# 之后可在任意目录使用：
lorahub doctor                # 查看环境健康度
lorahub                       # 打开 TUI（交互终端下）
lorahub --no-tui --help       # 传统命令帮助
lorahub service start         # 启动后台守护（随机端口）
lorahub service start --port 18765  # 指定端口
lorahub service status        # 查看运行状态
lorahub service stop          # 停止
lorahub service enable        # 注册系统级开机自启（Linux/macOS，需 sudo）
lorahub manage update         # 拉最新代码 + 重装依赖 + 重建前端
lorahub manage upgrade        # 切到最新 release tag
lorahub manage build          # 仅重建前端
lorahub --lang en --help      # 切换为英文 help / 输出
```

完整子命令见 `scripts/README.md`。

### 手动安装

```bash
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev,cpu]"
cd web && npm ci && cd ..
```

### Docker

一份 `Dockerfile` 同时服务 GPU 训练与 CPU 仅 API/管理两种形态，所有用户数据通过命名卷持久化，`docker compose down` 不丢任务历史与配置：

```bash
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
cp docker/.env.example docker/.env          # 可选：填镜像源 / token
docker compose -f docker/docker-compose.yml --profile gpu up -d --build
# CPU 机器：
docker compose -f docker/docker-compose.yml --profile cpu up -d --build
```

访问 `http://127.0.0.1:18765`。Docker 首次启动会在持久化卷生成访问令牌，读取命令见 [Docker 说明](docker/README.md)。镜像只携带应用本体，训练后端(kohya / diffusion-pipe / anima_lora / ai-toolkit)启动后在挂载卷里通过 Web UI 或 `lorahub bootstrap-kohya` 安装，与本地哲学一致。完整说明见 [docs/getting-started/docker.md](docs/getting-started/docker.md)。

### 训练后端

训练后端在 Web UI「设置 → 安装与升级」中安装。该流程会按当前 CUDA / 驱动给出可选 PyTorch 版本，并使用已配置的 GitHub / PyPI / PyTorch / npm 镜像。

```bash
lorahub bootstrap-kohya              # CLI 仍保留 kohya 安装入口
```

`anima_lora` 位于 `external/anima_lora/`，`ai-toolkit` 位于 `external/ai_toolkit/`，无需再 clone。首次安装后可在 Web UI 下载 Anima / Krea2 等基础模型；下载任务支持镜像、状态持久化与续传。

如需指向已有的 checkout：

```bash
export LORAHUB_KOHYA_SD_SCRIPTS=/path/to/sd-scripts
export LORAHUB_DIFFUSION_PIPE=/path/to/diffusion-pipe
export LORAHUB_ANIMA_LORA_PYTHON=/path/to/anima_lora/.venv/bin/python
export LORAHUB_AI_TOOLKIT_REPO=/path/to/ai-toolkit
```

### 可选依赖

| Extra     | 用途                                       |
| --------- | ------------------------------------------ |
| `api`     | FastAPI 服务（`lorahub serve`）            |
| `cpu`     | WD14 标注通过 CPU 版 `onnxruntime`         |
| `gpu`     | WD14 标注通过 `onnxruntime-gpu` 走 CUDA    |
| `tagging` | JoyTag（PyTorch）标注后端                  |
| `dev`     | 测试、lint、mypy、httpx                    |
| `docs`    | mkdocs 文档站                              |

`cpu` 与 `gpu` 不可同时安装；切换到 CUDA 版前先执行 `pip uninstall onnxruntime`。

---

## 快速开始

```bash
lorahub init my_character --template anima_lora_default
$EDITOR my_character.yaml                       # 填 checkpoint 与数据集路径
lorahub validate my_character.yaml              # 校验配置
lorahub train    my_character.yaml              # 启动训练
lorahub serve    --port 18765                   # 启动 Web UI + REST API（可选）
```

启动后访问 `http://127.0.0.1:18765`。Job 详情页通过 SSE 实时推送事件，断线后通过 `Last-Event-ID` 续传。

完整使用流程见 [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)。

---

## 界面预览

| 数据面板 | 训练配置 |
| --- | --- |
| <img src="docs/assets/screenshots/dashboard.png" alt="数据面板" width="420"> | <img src="docs/assets/screenshots/training-configs.png" alt="训练配置" width="420"> |
| 数据集 | 图像工作台 |
| <img src="docs/assets/screenshots/datasets.png" alt="数据集" width="420"> | <img src="docs/assets/screenshots/image-workbench.png" alt="图像工作台" width="420"> |
| 终端 | 参数搜索 |
| <img src="docs/assets/screenshots/terminal.png" alt="终端" width="420"> | <img src="docs/assets/screenshots/param-search.png" alt="参数搜索" width="420"> |

---

## 核心特性

- **后端独立配置表单**：kohya、diffusion-pipe、anima_lora、ai-toolkit 各自显示自己的字段，避免无关参数混在一起。
- **训练调度**：队列、取消、强制终止、恢复、异常事件、GPU slot、单任务多 GPU 分布式训练。
- **多 GPU 策略**：一任务一 GPU、单任务 DDP、Anima FSDP、Anima DeepSpeed ZeRO。
- **图像工作台**：导入、审计、WD14 / JoyTag 打标、智能 caption、去重、质量筛选、导出、LoRA 生图测试。
- **训练分析**：损失曲线、检查点、样图、事件时间线、AI 分析、任务对比。
- **模型下载与产物下载**：ModelScope / Hugging Face 镜像、状态持久化、续传、训练产物归档下载。
- **维护更新**：正式版 / dev 版本检测、更新说明、依赖重装、前端构建、按上次端口重启。
- **CLI / TUI / API**：命令行、交互式 TUI、REST + SSE 使用同一套任务与状态数据。

---

## 训练后端

| 后端             | 上游                                                                    | 覆盖范围                                                                | 备注                                              |
| ---------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| `kohya`          | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)           | SD1.5、SDXL、SD3、FLUX、Lumina、HunyuanImage、Anima                    | 适合经典 LoRA / LoCon / LoHa / DoRA 工作流。      |
| `diffusion-pipe` | [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) | Wan、HunyuanVideo、LTX、Cosmos、Flux2、Chroma、Qwen-Image 等            | DeepSpeed / pipeline 训练后端。                   |
| `anima_lora`     | [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora)         | Anima DiT                                                               | vendored 在 `external/anima_lora/`。              |
| `ai_toolkit`     | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)               | Krea2                                                                  | vendored 在 `external/ai_toolkit/`。              |

每个后端的字段映射在 `lorahub/core/backends/<name>/compiler.py`。前端配置表单按后端拆分，不再展示无关字段。

### 关于 vendored 的 anima_lora

`external/anima_lora/` 是 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 的固定 snapshot，按上游 license 随附。LoraHub 的魔改层提供自动 VAE / TE 预处理、val_loss、采样保护、V100 fp16 稳定化、全量微调、OrthoLoRA / T-LoRA / Hydra / LyCORIS 系列算法、postfix、EasyControl、IP-Adapter、DMD turbo、DDP / FSDP / ZeRO 分布式启动。

`external/` 下的 vendored 后端不要直接当作上游项目提交；LoraHub 集成与魔改代码主要在 `lorahub/core/backends/*/`。

---

## CLI / API

```bash
lorahub jobs ls                         # 列任务
lorahub jobs cancel <id>                # 优雅停止
lorahub jobs kill <id>                  # SIGKILL 进程组
lorahub jobs resume <id>                # 从最新 checkpoint 续训
lorahub jobs rerun <id>                 # 用同一份配置重跑

lorahub sweeps submit configs/sweep.yaml
lorahub sweeps ls

lorahub system gpu                      # CUDA 设备快照
lorahub system info                     # python / torch / 后端探测
```

REST 端点全部挂在 `/api`：`/api/configs`、`/api/jobs`、`/api/jobs/{id}/sse`、`/api/sweeps`、`/api/image-studio/*`、`/api/system/sse`、`/api/models/download` 等。默认绑定 `127.0.0.1`；非回环访问必须提供 `LORAHUB_API_TOKEN` 或使用自动生成的令牌。公网部署仍需 TLS。

---

## 许可证

LoraHub 采用 **AGPL-3.0-or-later**。完整文本见 [LICENSE](LICENSE)。如果你修改 LoraHub 并通过网络对外提供服务，AGPL 要求把修改后的源码同时开放给这些用户。

`external/anima_lora/` 与 `external/ai_toolkit/` 由各自上游 license 管辖，详见对应目录内的 `LICENSE`。

---

## 贡献

PR 与 issue 流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。社区行为规范见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

---

## 社区

本项目已在 [LINUX DO 社区](https://linux.do) 发布，感谢社区的支持与反馈。

---

## 致谢

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora)
- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- [SmilingWolf WD14 / WD-v3 taggers](https://huggingface.co/SmilingWolf)
- [fpgaminer/joytag](https://github.com/fpgaminer/joytag)
- [Optuna](https://optuna.org/)
- [FastAPI](https://fastapi.tiangolo.com/) / [Pydantic](https://docs.pydantic.dev/) / [typer](https://typer.tiangolo.com/) / [rich](https://rich.readthedocs.io/)
- [React](https://react.dev/) / [Vite](https://vitejs.dev/) / [TanStack Query](https://tanstack.com/query)
