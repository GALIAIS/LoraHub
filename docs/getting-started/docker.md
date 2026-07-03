---
title: Docker 部署
description: 用 Docker / Docker Compose 一键运行 LoraHub,含 GPU 与 CPU 两种形态。
---

# Docker 部署

LoraHub 提供官方 Docker 镜像构建方式,一份 `Dockerfile` 同时服务 **GPU 训练**与 **CPU 仅 API/管理** 两种场景。所有用户数据通过命名卷持久化,`docker compose down` 不会丢失。

> 镜像只携带 LoraHub 应用本体(代码 + 已构建前端 + 主 venv)。训练后端(kohya / diffusion-pipe / anima_lora / ai-toolkit)在容器启动后通过 Web UI 或 CLI 在挂载卷里 bootstrap——这与本地安装的哲学一致:每个后端维护自己隔离的 venv,版本可热换,镜像体积可控。

---

## 前置条件

- Docker Engine 24+ 与 Docker Compose v2。
- **GPU 形态**:NVIDIA 驱动 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)。装好后 `docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi` 应能列出 GPU。
- **CPU 形态**:无需任何 GPU 组件。

---

## 快速开始

```bash
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub

# (可选)复制环境变量文件,按需填写镜像源 / token
cp docker/.env.example docker/.env

# 构建并启动 GPU 形态
docker compose -f docker/docker-compose.yml --profile gpu up -d --build
```

启动后访问 `http://127.0.0.1:18765`。健康检查:

```bash
curl http://127.0.0.1:18765/api/health
```

CPU 机器改用:

```bash
docker compose -f docker/docker-compose.yml --profile cpu up -d --build
```

---

## 镜像携带 vs 卷持久化

理解这条边界,才能正确备份与迁移。

### 镜像携带(只读层,重建不变)

- LoraHub 主体代码(`lorahub/`、`pyproject.toml`、`configs/` 内置模板)。
- `external/anima_lora/`、`external/ai_toolkit/` 的 vendored 源码。
- 预构建的 `web/dist`(构建阶段 `npm run build` 产出)。
- 主 venv,含 `[api, gpu, tagging]` extras。

### 卷持久化(`/data` 命名卷,跨容器存活)

| 容器路径 | 内容 |
| --- | --- |
| `/data/runs/` | 任务历史 SQLite、训练产物、`events.jsonl` |
| `/data/configs/` | 用户训练配置 YAML |
| `/data/datasets/` `/data/models/` `/data/workspaces/` `/data/samples/` `/data/checkpoints/` | 用户图片、基础模型、训练产物 |
| `/data/.lorahub/` | 后端独立 venv(kohya / diffusion-pipe / anima / ai-toolkit)+ uv / python 工具链 |
| `/data/.cache/uv` | uv 包缓存(加速后端重装) |
| `/data/hf-home/hub` | HuggingFace 模型缓存(WD14 / JoyTag 首次加载) |
| `/data/xdg/data` | `settings.json` |
| `/data/xdg/state` | `tasks.sqlite3`、uvicorn 绑定/PID、更新检查缓存 |
| `external/anima_lora/.venv` | anima_lora 的 venv(单独命名卷,覆盖只读源码层) |
| `external/ai_toolkit/venv` | ai-toolkit 的 venv(同上) |

`settings.json` 与 `tasks.sqlite3` 原本经 `platformdirs` 落在 `~/.local/share` 与 `~/.local/state`,容器内会被 `XDG_DATA_HOME` / `XDG_STATE_HOME` 重定向进 `/data/xdg/`,与 `runs/` 一起被卷覆盖。HuggingFace 缓存原默认 `~/.cache/huggingface`,由 `HF_HOME` 重定向进 `/data/hf-home`。**这些重定向都在 `entrypoint.sh` 里完成,不改任何 Python 源码。**

### `down` 的行为

- `docker compose down`:**保留**命名卷,上述全部数据存活,`up` 即恢复。
- `docker compose down -v` 或 `docker volume rm lorahub-data lorahub-anima-venv lorahub-aitoolkit-venv`:**清空全部用户数据**(任务历史、配置、模型、后端 venv)。谨慎使用。

---

## GPU 配置

默认 GPU service 通过 `deploy.resources.reservations.devices` 暴露全部 NVIDIA GPU。限制数量:

```bash
# .env
LORAHUB_GPU_COUNT=1          # 只挂 1 块
# 或指定多块: 2, all(默认)
```

指定某几块 GPU 需改 `docker-compose.yml` 里的 `devices` 段为 `device_ids: ["0", "2"]`。

容器内 `nvidia-smi` 应可见 GPU;`lorahub system gpu`(经 `docker compose exec`)输出 CUDA 设备快照。WD14 标注自动走 `onnxruntime-gpu`。

---

## 安装训练后端

镜像不预装后端 venv。首次进 Web UI「设置 → 安装与升级」点击对应后端的「安装」按钮,bootstrap 跑在 `/data/.lorahub/` 卷里,产物持久化。

也可在容器内走 CLI:

```bash
# kohya(约 10 分钟:clone + venv + PyTorch + 依赖)
docker compose exec lorahub lorahub bootstrap-kohya

# 或指定 CUDA / torch 版本
docker compose exec lorahub lorahub bootstrap-kohya --cuda cu124 --torch 2.6.0
```

anima_lora 需 Python 3.13 + torch nightly,其 `uv sync` 会自动拉取,venv 落在 `external/anima_lora/.venv` 命名卷。ai-toolkit 同理落在 `external/ai_toolkit/venv`。

国内网络建议在 `docker/.env` 填镜像源,bootstrap 会自动走:

```bash
LORAHUB_GH_PROXY=https://gh-proxy.org/
LORAHUB_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
```

---

## 数据卷管理

### 用宿主机目录替代命名卷

若数据集 / 模型已在宿主机,改用 bind mount 更直接。编辑 `docker-compose.yml` 的 `volumes` 段:

```yaml
volumes:
  - lorahub-data:/data
  - /path/to/your/datasets:/data/datasets
  - /path/to/your/models:/data/models
```

### 备份

```bash
# 备份整个 /data(任务历史 + 配置 + 模型 + 后端 venv,体积可能很大)
docker run --rm -v lorahub-data:/data -v "$PWD":/backup \
  ubuntu tar czf /backup/lorahub-data-$(date +%F).tar.gz /data

# 仅备份任务历史与配置(轻量)
docker run --rm -v lorahub-data:/data -v "$PWD":/backup \
  ubuntu tar czf /backup/lorahub-light-$(date +%F).tar.gz \
  /data/runs /data/configs /data/xdg
```

### 迁移到另一台机器

```bash
# 旧机:备份(同上)
# 新机:构建镜像 → 恢复
docker run --rm -v lorahub-data:/data -v "$PWD":/backup \
  ubuntu tar xzf /backup/lorahub-data-2026-07-03.tar.gz -C /
docker compose -f docker/docker-compose.yml --profile gpu up -d
```

---

## 文件权限

镜像以非 root 用户 `lorahub` 运行(UID/GID 由 `PUID`/`PGID` 控制,默认 1000)。`entrypoint.sh` 启动前把 `/data` 的属主改成 `PUID:PGID`,使卷内文件与宿主机用户对齐。若 bind mount 的宿主机目录属主不是 1000,在 `.env` 设:

```bash
PUID=1002        # 改成宿主机用户的 UID(id -u)
PGID=1002        # 同上(id -g)
```

---

## 远程访问与安全

LoraHub 默认绑定 `127.0.0.1:18765`,**无内置鉴权**。如需远程访问:

1. 保持 `LORAHUB_BIND_ADDR=127.0.0.1`(默认)。
2. 用反向代理(Nginx / Caddy / Tailscale Funnel 等)加 TLS + 基本认证或 OAuth。
3. 反代需支持 SSE / WebSocket 长连接(任务事件流、系统遥测)。Nginx 关键配置:`proxy_buffering off;`、`proxy_read_timeout 1h;`。

**不要**直接把端口 publish 到公网。

---

## 常见问题

### `docker compose up` 后访问 `127.0.0.1:18765` 打不开

1. `docker compose logs lorahub` 看是否启动失败(常见:GPU 工具链未装、卷权限)。
2. `docker compose exec lorahub curl -fsS http://127.0.0.1:18765/api/health` 验证容器内是否通。
3. 首次启动需装 Python 依赖,`start_period` 给了 30s;冷启动慢的话等一会再看。

### WD14 标注第一次很慢 / 模型重复下载

模型首次加载会从 HuggingFace 拉取并缓存到 `/data/hf-home/hub`。确认 `HF_HOME` 没被覆盖、卷没被 `down -v` 清掉。国内可设 `HF_ENDPOINT=https://hf-mirror.com`。

### 容器内看不到 GPU

确认 NVIDIA Container Toolkit 已装且 `docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi` 在宿主机能跑。Windows + WSL2 需在 WSL2 内装 toolkit。

### 后端 bootstrap 失败:网络超时

在 `docker/.env` 填镜像源(见上文「安装训练后端」),`docker compose up -d` 重建后重试。`LORAHUB_GH_PROXY` 影响 git clone 与 uv release 下载,`LORAHUB_PYPI_INDEX` / `UV_DEFAULT_INDEX` 影响 pip/uv 装包。

### `external/anima_lora/.venv` 卷与镜像源码版本不一致

镜像升级后 vendored 源码变了,但旧 venv 卷还在。进 Web UI「设置 → 安装」点 anima_lora 的「重装」,或:

```bash
docker compose run --rm --entrypoint bash lorahub -c "rm -rf /app/external/anima_lora/.venv"
docker compose up -d   # 再进 UI 触发安装
```

---

## 构建参数

`docker/Dockerfile` 支持以下 `--build-arg`(也可在 `.env` 里以同名变量提供,compose 自动透传):

| 参数 | 用途 |
| --- | --- |
| `LORAHUB_PYPI_INDEX` | PyPI 镜像(影响主 venv 与后端 bootstrap) |
| `LORAHUB_GH_PROXY` | GitHub 代理前缀(uv 二进制、git clone) |
| `NPM_CONFIG_REGISTRY` | npm 镜像(前端构建) |

例:

```bash
docker build -t lorahub:dev \
  --build-arg LORAHUB_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg NPM_CONFIG_REGISTRY=https://registry.npmmirror.com \
  -f docker/Dockerfile .
```
