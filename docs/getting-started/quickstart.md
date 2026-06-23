---
title: 快速开始
description: 起一份配置并跑出第一次训练。
---

# 快速开始

LoraHub 装完、后端在位之后，从零到一次训练只要四条命令。

```powershell
# 1. 起一份配置
lorahub init my_character

# 2. 编辑 configs/my_character.yaml：填 checkpoint 路径与数据集路径
notepad configs/my_character.yaml

# 3. 干跑校验（不会真训练）
lorahub validate configs/my_character.yaml
lorahub info     configs/my_character.yaml

# 4. 训练
lorahub train    configs/my_character.yaml
```

## 一份最小配置

YAML 在 wire 上用 camelCase；validator 兼容旧版的 snake_case，老配置不会失效。

```yaml
schemaVersion: "1.0"
baseModel:
  arch: sdxl
  checkpoint: ./models/sdxl_base_1.0.safetensors
dataset:
  source: ./datasets/my_character
  resolution: [1024, 1024]
network:
  type: lora
  rank: 32
  alpha: 16
schedule:
  epochs: 10
  batchSize: 1
  gradAccum: 4
precision: bf16
gradientCheckpointing: true
output:
  name: my_character_v1
backend:
  type: kohya
```

完整带注释的范例见 [`configs/anima_lora_default.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/anima_lora_default.yaml)(上游 `make lora default` 的 100% 复刻基线);8GB 显存档见 [`configs/anima_lora_8gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/anima_lora_8gb.yaml),32GB 画风高吞吐档见 [`configs/anima_style_32gb_loha.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/anima_style_32gb_loha.yaml)。

## 自动按机器调参

`lorahub init --auto` 会探测 `nvidia-smi` 拿到显存大小，扫描数据集目录数图片，从 checkpoint 文件名识别架构，然后写一份 rank / batch / grad_accum 按显存档位调好的配置：

```powershell
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

`--vram-mib 8192` 可手动覆盖显存检测。

## `lorahub info` 显示什么

`lorahub info` 是 dry-run：把配置编译成将要 launch 的后端 argv（kohya CLI flags 或 diffusion-pipe TOML），打印 entry script，估算显存峰值，**不动 GPU**。开长跑前过一眼。

## 启动 Web UI

CLI 训练之外，Web 工作台覆盖了完整的可视化操作（数据集浏览、配置编辑、Job 队列、样本图、Sweep）：

```powershell
lorahub serve --port 18765
```

或者用一键脚本：

```powershell
scripts\run.bat                # 默认 prod：API + 已构建的 SPA
scripts\run.bat dev            # dev：API + Vite 热更新
```

启动后访问 `http://127.0.0.1:18765`。Job 详情页通过 SSE 实时推送事件，断线后通过 `Last-Event-ID` 续传。

## 下一步

- [端到端教程](smoke-test.md) — 从 BangumiBase 图片到训练好的 LoRA 全流程。
- [配置字段参考](../configs/fields.md) — schema 的每一个旋钮。
