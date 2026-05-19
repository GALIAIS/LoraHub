---
title: HTTP API
description: LoraHub 工作台的 REST + SSE + WebSocket 接口面。
---

# HTTP API

LoraHub 自带 FastAPI 服务器供编程访问。装 API extras 起服务:

```powershell
pip install lorahub[api]
lorahub serve --port 18765
```

默认绑 `127.0.0.1`,无认证 — 仅适用于本机。Job 元数据持久化到
`runs/jobs.sqlite`,事件 ring 保持进程内。同级 store:`runs/ai.sqlite`
(AI providers + routes)、`runs/image_studio.sqlite`(annotations + phash
+ pending ops)、`runs/sweeps.sqlite`(sweep plans + trials,TPE study 落
盘走这里)、`runs/sessions.sqlite`(长任务 session handle)。

## 布局

- 所有 API 路由挂 `/api`。
- 站点根与 `/{spa-path}` 留给 React 前端,从 `web/dist` 静态挂载。
- 业务路由按域分文件:`lorahub.api.routers.*`,一资源一路由。`image_studio`
  在 B7 后已拆成 9 个子模块(`listings / annotations / ops / ai / datasets /
  dedupe / similarity / tagging` + `_shared`),URL 不变。
- 实时通道首选 SSE(`/api/.../sse`,带 `Last-Event-ID` 续传);WebSocket
  (`/api/.../stream`)保留作 fallback。新客户端默认走 SSE。

## 一键 launcher

`scripts/` 下有跨平台 launcher,自动解析项目 venv、补依赖、并起 API + Vite:

=== "Windows"

    ```powershell
    scripts\launch.bat              # 默认: dev (API + Vite)
    scripts\launch.bat -Mode prod   # 仅 API, 服务预构建的 web/dist
    scripts\launch.bat -Mode build  # 一次性 npm install + vite build
    ```

=== "macOS / Linux"

    ```bash
    chmod +x scripts/launch.sh
    scripts/launch.sh                       # 默认 dev
    scripts/launch.sh --mode prod --port 8080
    scripts/launch.sh --mode build
    ```

launcher 自动认 `.venv/` 和 `web/node_modules/`,只在缺东西时跑
`pip install -e ".[api,dev]"` 与 `npm install`,并把 Vite 的 `/api` 代理转发
到 API 实际端口。`--no-install` 跳依赖检查,`--reload` 开 uvicorn 自动重载。

## 读 OpenAPI

服务器在线时 FastAPI 把 OpenAPI 文档挂在 `/openapi.json`,Swagger UI 在
`/docs`。这两个是权威源 — [Endpoints](endpoints.md) 页是接口面的速查。
