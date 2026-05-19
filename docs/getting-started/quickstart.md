---
title: 快速开始
description: 起一份配置并跑出第一次训练。Scaffold a config and run your first training job.
---

# 快速开始

LoraHub 装完、后端在位之后,从零到一次训练只要四条命令。

```powershell
# 1. 起一份配置
lorahub init my_character

# 2. 编辑 configs/my_character.yaml: 指向你的 checkpoint 和数据集
notepad configs/my_character.yaml

# 3. 干跑校验(不会真训练)
lorahub validate configs/my_character.yaml
lorahub info     configs/my_character.yaml

# 4. 训练
lorahub train    configs/my_character.yaml
```

## 一份最小配置

YAML wire 用 camelCase(validator 仍兼容旧的 snake_case,老配置不会
失效):

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

完整带注释的例子见 [`configs/sdxl_character_8gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/sdxl_character_8gb.yaml);
diffusion-pipe 路径上的 Anima 配置见
[`configs/anima_style_24gb.yaml`](https://github.com/GALIAIS/LoraHub/blob/main/configs/anima_style_24gb.yaml)
和 `configs/anima_character_24gb.yaml`。

## 自动按机器调参

`lorahub init --auto` 会探测 `nvidia-smi` 拿到显存,扫数据集目录数图片,
从 checkpoint 文件名识别架构,然后写一份 rank / batch / grad_accum 按
显存档位调好的配置:

```powershell
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

`--vram-mib 8192` 可手动覆盖显存检测。

## `lorahub info` 显示什么

`lorahub info` 是 dry-run:把配置编译成将要 launch 的 backend argv
(kohya CLI flags 或 diffusion-pipe TOML),打印 entry script,估算显存
峰值,**不动 GPU**。开长跑前过一眼很有用。

## 下一步

- [冒烟测试](smoke-test.md) — 从 BangumiBase 图片到训练好的 LoRA 全流程。
- [配置字段参考](../recipes/fields.md) — schema 的每一个旋钮。

---

## English

After `pip install` and a backend bootstrap, the path from zero to a
running job is four commands:

```powershell
lorahub init my_character
notepad configs/my_character.yaml
lorahub validate configs/my_character.yaml
lorahub info     configs/my_character.yaml
lorahub train    configs/my_character.yaml
```

The minimal config above describes a SDXL LoRA training run; YAML on
the wire is camelCase, and the validator also accepts the legacy
snake_case so older recipes keep loading.

`lorahub init --auto` autotunes per machine: it probes `nvidia-smi` for
VRAM, scans the dataset folder, infers the architecture from the
checkpoint filename, and writes a config with rank / batch /
grad_accum chosen for the detected VRAM tier. Use `--vram-mib N` to
override detection.

`lorahub info` is a dry run — it compiles the config to backend argv,
prints the entry script, and reports an estimated VRAM peak without
touching the GPU. Run it before a long training session.
