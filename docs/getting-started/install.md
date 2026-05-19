---
title: 安装
description: 安装 LoraHub 并 bootstrap 一个训练后端。Install LoraHub and bootstrap a training backend.
---

# 安装

需要 Python 3.11 或 3.12,以及一张显存 ≥ 8 GB 的 NVIDIA GPU。至少要装一个
训练后端:

- `kohya-ss/sd-scripts` — 覆盖 SD / SDXL / Flux / Lumina / HunyuanImage / Anima。
- `tdrussell/diffusion-pipe` — 覆盖现代 DiT 阵列(Flux2 / Chroma / Wan /
  Cosmos / Anima 等)。
- `sorryhyun/anima_lora` — Anima 专用栈,LoraHub 已 vendored 在
  `external/anima_lora/`,无需 clone。

三者可共存,通过每份配置的 `backend.type` 字段切换。

## 取源码

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
```

## Bootstrap 一个后端

=== "Option A: 在工作树内 bootstrap"

    ```powershell
    # kohya — SD / SDXL / Flux / Lumina / HunyuanImage / Anima
    lorahub bootstrap-kohya              # 约 10 分钟: clone + venv + PyTorch + 依赖
    # diffusion-pipe — DiT 阵列
    lorahub bootstrap-diffusion-pipe
    ```

    `bootstrap-kohya` 默认 PyTorch 2.6.0 + CUDA 12.4。可用 `--cuda cu121`
    (或 `cu118` / `cu128`)、`--torch 2.6.0` 切版本,`--no-xformers` 跳过
    可选 xformers,`--force` 抹掉半装好的目录重来。

=== "Option B: 指向已有 checkout"

    ```powershell
    $env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
    $env:LORAHUB_DIFFUSION_PIPE   = "C:\path\to\diffusion-pipe"
    # 或者把 .env.example 复制成 .env 后编辑
    ```

!!! tip ".env 自动加载"
    LoraHub 启动时会从项目根读取 `.env`,所以一旦 `.env` 里写了
    `LORAHUB_KOHYA_SD_SCRIPTS=./sd-scripts`,就不必每次 shell 都 export。

## 可选 extras

| Extra     | 何时安装                                       | 命令                                  |
| --------- | ---------------------------------------------- | ------------------------------------- |
| `api`     | FastAPI 服务器(`lorahub serve`)              | `pip install -e ".[api]"`             |
| `gpu`     | WD14 tagger 走 CUDA(`onnxruntime-gpu`)       | `pip install -e ".[gpu]"`             |
| `tagging` | JoyTag(PyTorch)tagger backend                | `pip install -e ".[tagging]"`         |
| `dev`     | 测试、lint、mypy、httpx                        | `pip install -e ".[dev]"`             |
| `docs`    | 构建本站文档                                   | `pip install -e ".[docs]"`            |

## anima_lora venv

anima_lora 后端要 PyTorch 2.11/2.12 nightly + CUDA 13.x,LoraHub 主 venv
通常装不进去。建议在 `external/anima_lora/` 旁边单建一份 venv 并 `uv sync`,
然后用 `LORAHUB_ANIMA_LORA_PYTHON` 指过去:

```powershell
$env:LORAHUB_ANIMA_LORA_PYTHON = "C:\path\to\anima_lora\.venv\Scripts\python.exe"
```

LoraHub 自身不会 import anima_lora 的代码,只把它作为子进程拉起。

## Anima Base 下载器

LoraHub 自带一个 Anima 全栈下载脚本(约 5.5 GB,transformer + Qwen-Image
VAE + Qwen3-0.6B 文本编码器):

```powershell
bash scripts/_download_anima.sh        # 默认走 hf-mirror.com,国内可用
```

文件落到 `models/circlestone-labs__Anima/split_files/`。`configs/anima_style_24gb.yaml`
和 `configs/anima_character_24gb.yaml` 直接指向这套布局。

## 下一步

- [快速开始](quickstart.md) — 四条命令出第一份 config。
- [冒烟测试](smoke-test.md) — 用真实图片跑通完整流水线。

---

## English

LoraHub needs Python 3.11 or 3.12 and an NVIDIA GPU with at least 8 GB
of VRAM. Pick at least one training backend:

- `kohya-ss/sd-scripts` — SD / SDXL / Flux / Lumina / HunyuanImage / Anima.
- `tdrussell/diffusion-pipe` — the modern DiT roster (Flux2, Chroma,
  Wan, Cosmos, Anima, ...).
- `sorryhyun/anima_lora` — Anima-specific stack, vendored at
  `external/anima_lora/` so no extra clone is needed.

All three may coexist; switch per-config via `backend.type`.

### Get the source

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
```

### Bootstrap

`lorahub bootstrap-kohya` and `lorahub bootstrap-diffusion-pipe` clone
the upstream, build a venv, and install PyTorch + deps. Defaults are
PyTorch 2.6.0 + CUDA 12.4; pass `--cuda cu121|cu118|cu128`,
`--torch X.Y.Z`, `--no-xformers`, or `--force` to override. To use an
existing checkout, set `LORAHUB_KOHYA_SD_SCRIPTS` and
`LORAHUB_DIFFUSION_PIPE` (or copy `.env.example` to `.env`). LoraHub
auto-loads `.env` from the project root.

The `anima_lora` backend is vendored — point
`LORAHUB_ANIMA_LORA_PYTHON` at a venv that has the required
nightly PyTorch + CUDA 13.x, and LoraHub will spawn it as a
subprocess (it never imports anima_lora code in-process).

### Optional extras

`api`, `gpu`, `tagging`, `dev`, `docs` — same matrix as the table
above. License: AGPL-3.0-or-later.
