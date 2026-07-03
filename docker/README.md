# LoraHub Docker

一份 `Dockerfile` 同时服务 GPU 与 CPU 两种形态,所有用户数据通过命名卷持久化。

> 完整文档见 [docs/getting-started/docker.md](../docs/getting-started/docker.md)。本文件是速查卡。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `docker/Dockerfile` | 多阶段构建:node 构建前端 → nvidia/cuda 运行时装 Python+uv+`[api,gpu,tagging]` extras |
| `docker/docker-compose.yml` | `gpu` / `cpu` 两个互斥 profile,命名卷,127.0.0.1 端口 |
| `docker/entrypoint.sh` | 非 root 降权、XDG/HF 缓存重定向、`exec uvicorn --host 0.0.0.0` |
| `docker/.env.example` | 镜像源 / token / UID 等占位 |
| 仓库根 `.dockerignore` | 排除本地状态/用户数据(Docker 要求放在 context 根,即 repo root) |

## 快速开始

```bash
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
cp docker/.env.example docker/.env          # 可选:按需填镜像源

# GPU
docker compose -f docker/docker-compose.yml --profile gpu up -d --build

# CPU
docker compose -f docker/docker-compose.yml --profile cpu up -d --build
```

访问 `http://127.0.0.1:18765`。

## 数据持久化

所有状态在命名卷 `lorahub-data` → 容器 `/data`(`LORAHUB_HOME`):

- `runs/` 任务历史 + 训练产物
- `configs/` `datasets/` `models/` `workspaces/` `samples/` `checkpoints/` 用户数据
- `.lorahub/` 后端 venv + 工具链
- `hf-home/hub` HuggingFace 模型缓存
- `xdg/data` `xdg/state` settings/tasks/runtime 绑定

`docker compose down` 保留卷;`down -v` 清空全部数据。

## 安装训练后端

镜像不预装后端。启动后进 Web UI「设置 → 安装与升级」点击安装,或:

```bash
docker compose exec lorahub lorahub bootstrap-kohya
```

产物落在 `/data/.lorahub/` 卷,持久化。

## 远程访问

默认 `127.0.0.1`,**无内置鉴权**。远程访问请用反代加 TLS + 认证,勿直接 publish 到公网。详见完整文档。
