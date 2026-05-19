---
title: 配置模板
description: 内置起步模板与填充用 placeholder 体系。
---

# 配置模板

LoraHub 自带几份 YAML 模板,放在
[`configs/`](https://github.com/GALIAIS/LoraHub/tree/main/configs)(旧的
`configs/builtin/` 子目录已提到顶层)。Web UI 的配置向导通过
`GET /api/configs/templates` 读它们,每份模板渲一张卡片,用户只填 placeholder
就能保存为 `configs/<name>.yaml`。

## 目录

| 模板 | 架构 | Network | 备注 |
| ---- | ---- | ------- | ---- |
| `sdxl_character_8gb` | SDXL | LoRA, rank 32 / alpha 16 | 8 GB 显存友好,1024 px,10 epoch,只训 UNet |
| `anima_style_24gb` | Anima(dp) | LoRA, rank 16 / alpha 8 | 24 GB / 4090 上的风格 LoRA;200 步一次 checkpoint;开 live preview |
| `anima_character_24gb` | Anima(dp) | LoRA, rank 32 / alpha 16 | 24 GB / 4090 上的角色 LoRA;200 步一次 checkpoint |

Anima 配置是 diffusion-pipe 路径的标准范例 — 把 transformer + Qwen-Image VAE +
Qwen3-0.6B 文本编码器接好,并打开 LoraHub 的 live preview worker。

## 默认 sample prompt 集

`configs/sample_prompts/anima_default.txt` 与两份 Anima recipe 一同发布 — 8 条
prompt 涵盖 portrait / cowboy shot / full body / group / scene / wide
landscape,内嵌 `@Kiko.L` trigger。直接改文件即可,live preview worker 在下一
个 checkpoint 时重读。

## Placeholder 格式

每份模板 YAML 可带两段顶层元信息(schema 校验 `TrainingConfig` 前会剥掉):

- `_template` — UI 卡片元信息(`name`、`description`、`arch`)。
- `_placeholders` — 用户实例化时必须填的字段列表,每项含 `key`、`label`、
  `path_field`、`placeholder`。

例:

```yaml
_placeholders:
  - key: name
    label: Config / output name
    path_field: output.name
    placeholder: my_character_v1
  - key: checkpoint
    label: SDXL base model checkpoint
    path_field: baseModel.checkpoint
    placeholder: C:\models\sdxl_base.safetensors
  - key: dataset
    label: Dataset directory
    path_field: dataset.source
    placeholder: ./datasets/my_character
```

向导每个 placeholder 渲一个表单字段,提交后
`POST /api/configs/templates/{template_id}/instantiate` 把值深度合并进模板,
校验后写到 `configs/<name>.yaml`。

## 加新模板

1. 把 YAML 丢进 `configs/`。
2. 加 `_template` 段(否则用文件 stem 作 name)。
3. `_placeholders` 列出每个必填路径或标签。
4. 重启 API 服务器。损坏模板会被记日志后跳过 — 单文件出错不影响目录加载。

## CLI 短路径

`lorahub init <name>` 从磁盘复制其中一份模板:

```powershell
lorahub init my_character                          # 默认: sdxl_character_8gb
lorahub init my_style --template anima_style_24gb  # 选 configs/anima_style_24gb.yaml
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character           # 按检测到的显存调参
```
