---
title: CLI
description: lorahub 命令行接口。
---

# CLI

LoraHub 提供一个 `lorahub` 入口,基于 [typer](https://typer.tiangolo.com/)
+ [rich](https://rich.readthedocs.io/)。每条命令要么作用于一份配置 YAML,
要么作用于由 `.env` 与工作台设置维护的全局状态。

## 命令分组

工作流命令:

| 命令 | 作用 |
| ---- | ---- |
| [`init`](commands.md#init) | 在 `configs/` 起一份起步配置。 |
| [`validate`](commands.md#validate) | 不跑训练,只校验配置。 |
| [`info`](commands.md#info) | dry-run:打印编译后的 argv + 估算显存。 |
| [`train`](commands.md#train) | 跑一次完整训练。 |
| [`sweep`](commands.md#sweep) | 规划或提交一次 sweep。 |
| [`serve`](commands.md#serve) | 拉起 FastAPI 服务器。 |
| [`version`](commands.md#version) | 打印当前版本。 |

数据 / 后端命令:

| 命令 | 作用 |
| ---- | ---- |
| [`bootstrap-kohya`](commands.md#bootstrap-kohya) | 一键装 kohya-ss/sd-scripts。 |
| [`bootstrap-diffusion-pipe`](commands.md#bootstrap-diffusion-pipe) | 一键装 tdrussell/diffusion-pipe。 |
| [`fetch-bangumi`](commands.md#fetch-bangumi) | 从 BangumiBase 下载单角色图集。 |
| [`tag`](commands.md#tag) | 对目录跑 WD14 / JoyTag 自动打标。 |
| [`caption normalize`](commands.md#caption-normalize) | 批量做 caption 变换。 |
| [`anima-caption`](commands.md#anima-caption) | Anima 格式 caption 重排。 |

运维命令(B9):

| 命令 | 作用 |
| ---- | ---- |
| [`jobs ls`](commands.md#jobs-ls) | 列训练任务。 |
| [`jobs show`](commands.md#jobs-show) | 看单 job 详情。 |
| [`jobs cancel`](commands.md#jobs-cancel) | 取消运行中的 job。 |
| [`jobs kill`](commands.md#jobs-kill) | SIGKILL 进程组,救援用。 |
| [`jobs resume`](commands.md#jobs-resume) | 从最近 checkpoint 续训。 |
| [`jobs rerun`](commands.md#jobs-rerun) | 原地重跑同一 job。 |
| [`sweeps submit`](commands.md#sweeps-submit) | 提交 sweep 计划到 API。 |
| [`sweeps ls`](commands.md#sweeps-ls) | 列 sweeps。 |
| [`system gpu`](commands.md#system-gpu) | 一次性 GPU 快照。 |
| [`system info`](commands.md#system-info) | host CPU / RAM / 磁盘 / 网络快照。 |

## 约定

- 任何命令都接 `--help`。
- 状态行用 ASCII(`OK`、`->`),Windows GBK 控制台也能正确显示。
- 启动时自动加载 cwd 的 `.env`,已存在的环境变量优先。
- Job 产物默认落 `runs/<output.name>/`;SQLite 元数据在 `runs/jobs.sqlite`,
  AI store 在 `runs/ai.sqlite`,Image Studio store 在
  `runs/image_studio.sqlite`,Sweep store 在 `runs/sweeps.sqlite`,长任务
  session 在 `runs/sessions.sqlite`。

完整逐命令参考见 [Commands](commands.md)。
