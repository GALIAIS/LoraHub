---
title: CLI
description: lorahub 命令行接口。
---

# CLI

LoraHub 提供一个 `lorahub` 入口，基于 [typer](https://typer.tiangolo.com/) + [rich](https://rich.readthedocs.io/)。每条命令要么作用于一份配置 YAML，要么作用于由 `.env` 与工作台设置维护的全局状态。

## 命令分组

工作流命令：

| 命令 | 作用 |
| ---- | ---- |
| [`init`](commands.md#init) | 在 `configs/` 下生成模板配置。 |
| [`validate`](commands.md#validate) | 不跑训练，只校验配置。 |
| [`info`](commands.md#info) | dry-run：打印编译后的 argv 与估算显存。 |
| [`train`](commands.md#train) | 跑一次完整训练。 |
| [`sweep`](commands.md#sweep) | 规划或提交一次 sweep。 |
| [`serve`](commands.md#serve) | 启动 FastAPI 服务。 |
| [`version`](commands.md#version) | 打印当前版本。 |

数据 / 后端命令：

| 命令 | 作用 |
| ---- | ---- |
| [`bootstrap-kohya`](commands.md#bootstrap-kohya) | 一键安装 kohya-ss/sd-scripts。 |
| [`bootstrap-diffusion-pipe`](commands.md#bootstrap-diffusion-pipe) | 一键安装 tdrussell/diffusion-pipe。 |
| [`fetch-bangumi`](commands.md#fetch-bangumi) | 从 BangumiBase 下载单角色图集。 |
| [`tag`](commands.md#tag) | 对目录跑 WD14 / JoyTag 自动打标。 |
| [`caption normalize`](commands.md#caption-normalize) | 批量做 caption 变换。 |
| [`anima-caption`](commands.md#anima-caption) | 把 caption 重排成 Anima 格式。 |

运维命令：

| 命令 | 作用 |
| ---- | ---- |
| [`jobs ls`](commands.md#jobs-ls) | 列训练任务。 |
| [`jobs show`](commands.md#jobs-show) | 看单 job 详情。 |
| [`jobs cancel`](commands.md#jobs-cancel) | 优雅取消运行中的 job。 |
| [`jobs kill`](commands.md#jobs-kill) | SIGKILL 进程组，救援用。 |
| [`jobs resume`](commands.md#jobs-resume) | 从最新 checkpoint 续训。 |
| [`jobs rerun`](commands.md#jobs-rerun) | 原地用同一份配置重跑。 |
| [`sweeps submit`](commands.md#sweeps-submit) | 把 sweep 计划提交到 API。 |
| [`sweeps ls`](commands.md#sweeps-ls) | 列 sweep 状态。 |
| [`system gpu`](commands.md#system-gpu) | 一次性 GPU 快照。 |
| [`system info`](commands.md#system-info) | host CPU / RAM / 磁盘 / 网络快照。 |

## 约定

- 所有命令都接 `--help`。
- 状态行使用 ASCII（`OK`、`->`），Windows GBK 控制台也能正确显示。
- 启动时自动加载 cwd 下的 `.env`，已存在的环境变量优先。
- Job 产物默认落 `runs/<output.name>/`；元数据 SQLite 见下表。

| 文件                           | 内容                                |
| ------------------------------ | ----------------------------------- |
| `runs/jobs.sqlite`             | Job 记录                            |
| `runs/ai.sqlite`               | AI provider / route                 |
| `runs/image_studio.sqlite`     | Image Studio 标注、phash、待办      |
| `runs/sweeps.sqlite`           | Sweep 计划与 trial（含 TPE study）  |
| `runs/sessions.sqlite`         | 长任务 session handle               |

完整的逐命令参考见 [命令参考](commands.md)。

