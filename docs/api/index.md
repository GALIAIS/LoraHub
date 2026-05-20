---
title: HTTP API
description: LoraHub 工作台的 REST + SSE + WebSocket 接口。
---

# HTTP API

LoraHub 自带 FastAPI 服务供编程访问。装好 API extras 后启动：

```powershell
pip install lorahub[api]
lorahub serve --port 18765
```

默认绑 `127.0.0.1`，无内置鉴权——仅用于本机。Job 元数据持久化到 `runs/jobs.sqlite`，事件 ring 保持在进程内。同级别的 store：

| 文件                       | 内容                                     |
| -------------------------- | ---------------------------------------- |
| `runs/jobs.sqlite`         | Job 记录                                  |
| `runs/ai.sqlite`           | AI provider 与 route                      |
| `runs/image_studio.sqlite` | 标注、phash、待办操作                     |
| `runs/sweeps.sqlite`       | Sweep plan 与 trial（含 TPE study）       |
| `runs/sessions.sqlite`     | 长任务 session handle                     |

## 布局

- 所有 API 路由挂在 `/api`。
- 站点根与 `/{spa-path}` 留给 React 前端，从 `web/dist` 静态挂载。
- 业务路由按域分文件：`lorahub.api.routers.*`，一个资源一个路由模块。`image_studio` 拆成 9 个子模块（`listings / annotations / ops / ai / datasets / dedupe / similarity / tagging` + `_shared`），URL 不变。
- 实时通道首选 SSE（`/api/.../sse`，带 `Last-Event-ID` 续传）；WebSocket（`/api/.../stream`）保留作 fallback。新客户端默认走 SSE。

## 一键启动器

`scripts/` 下有跨平台启动器，自动解析项目 venv、补依赖、并起 API + Vite：

=== "Windows"

    ```powershell
    scripts\run.bat              # dev：API + Vite
    scripts\run.bat prod         # prod：仅 API，服务预构建的 web/dist
    scripts\run.bat api          # 仅启动 API
    ```

=== "Linux / macOS"

    ```bash
    chmod +x scripts/run.sh
    scripts/run.sh               # dev
    scripts/run.sh prod
    scripts/run.sh api
    ```

启动器会优先用项目根的 `.lorahub/uv`、`.node/`、`.venv/`（一键安装脚本生成的本地工具链），再在缺失时回退系统 PATH。

## 读 OpenAPI

服务器在线时 FastAPI 在 `/openapi.json` 提供 OpenAPI 描述，Swagger UI 在 `/docs`。这两个是权威源——[端点](endpoints.md) 页是接口面的速查。

