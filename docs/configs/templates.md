---
title: 配置模板
description: 内置起步模板与填充用 placeholder 体系。
---

# 配置模板

LoraHub 自带几份 YAML 模板，放在 [`configs/`](https://github.com/GALIAIS/LoraHub/tree/main/configs)（旧的 `configs/builtin/` 子目录已提到顶层）。Web UI 的配置向导通过 `GET /api/configs/templates` 读它们，每份模板渲一张卡片，用户只填 placeholder 就能保存为 `configs/<name>.yaml`。

## 目录

| 模板 | 架构 | Network | 备注 |
| ---- | ---- | ------- | ---- |
| `anima_lora_default`        | Anima | LoRA + OrthoLoRA + T-LoRA, rank 16 / alpha 16 | 上游 anima_lora `make lora default` 的 100% 复刻基线;新手对照参考。 |
| `anima_lora_8gb`            | Anima | LoRA + OrthoLoRA + T-LoRA, rank 8 / alpha 8   | 8GB 显存档:768²、AdamW8bit、blocks_to_swap=24、grad-ckpt 开;关 sampling/validation/torch.compile。 |
| `anima_style_32gb_loha`     | Anima | LoHa + T-LoRA, rank 4 / alpha 4               | 32GB 卡上的画风 LoRA;LoHa 比同等表达的 LoRA 参数量减半,收敛更快。 |

Anima 配置走 LoraHub 的 anima_lora 后端 — 把 DiT + Qwen-Image VAE + Qwen3-0.6B 文本编码器接好,并自动调用上游 preprocess (resize_images / cache_latents / cache_text_embeddings)。

## 默认 sample prompt 集

`configs/sample_prompts/anima_default.txt` 与两份 Anima 配置一同发布——8 条 prompt 涵盖 portrait / cowboy shot / full body / group / scene / wide landscape，内嵌 `@Kiko.L` trigger。直接改文件即可，live preview worker 在下一个 checkpoint 时重读。

## Placeholder 格式

每份模板 YAML 可带两段顶层元信息（schema 校验 `TrainingConfig` 前会剥掉）：

- `_template` — UI 卡片元信息（`name`、`description`、`arch`）。
- `_placeholders` — 用户实例化时必须填的字段列表，每项含 `key`、`label`、`path_field`、`placeholder`。

示例：

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

向导每个 placeholder 渲一个表单字段，提交后 `POST /api/configs/templates/{template_id}/instantiate` 把值深度合并进模板，校验后写到 `configs/<name>.yaml`。

## 加新模板

1. 把 YAML 丢进 `configs/`。
2. 加 `_template` 段（否则用文件 stem 作 name）。
3. `_placeholders` 列出每个必填路径或标签。
4. 重启 API 服务。损坏模板会被记日志后跳过——单文件出错不影响目录加载。

## CLI 短路径

`lorahub init <name>` 从磁盘复制其中一份模板：

```powershell
lorahub init my_character                            # 默认:anima_lora_default
lorahub init my_8gb --template anima_lora_8gb        # 8GB 卡起步
lorahub init my_style --template anima_style_32gb_loha      # 32GB 画风 LoHA
lorahub init my_character --auto `
    --checkpoint .\models\circlestone-labs__Anima\split_files\diffusion_models\anima-base-v1.0.safetensors `
    --dataset    .\datasets\my_character             # 按检测到的显存调参
```
