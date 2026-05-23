# LoraHub recipes

LoraHub 自带的训练配方目录。每份 YAML 都已经端到端验证过 schema + backend.validate,可以直接 copy 后改 `dataset.source` / `output.name` 开训。

## 现有 6 份配方

| 文件 | 后端 | 算法 | 显存档 | 简介 |
|------|------|------|--------|------|
| `anima_lora_default.yaml` | anima_lora | LoRA + OrthoLoRA + T-LoRA | 通用 | 100% 复刻上游 anima_lora `make lora` 的三层 merge 链(base + lora + default preset)。新手对照参考的基线。 |
| `anima_lora_8gb.yaml` | anima_lora | LoRA + OrthoLoRA + T-LoRA | 8GB | 在 default 基础上压低显存预算:768²、AdamW8bit、blocks_to_swap=24、gradient_checkpointing on、关 sampling/validation/torch.compile。RTX 3060/4060 等 8GB 卡的安全档。 |
| `anima_character_32gb_dora.yaml` | anima_lora | DoRA + T-LoRA | 32GB | 角色 LoRA 高吞吐档:1024²、networkDim 32、batchSize 2 + gradAccum 4、torch.compile cudagraph_trees、sampling + CMMD 验证全开。RTX 4090 32G / 5090 / A6000 / A100 40G。 |
| `anima_style_32gb_loha.yaml` | anima_lora | LoHa + T-LoRA | 32GB | 画风 LoRA 高吞吐档,同硬件预算用 LoHa(rank=4 等效 dim=16,参数量减半收敛更快)代替 DoRA。 |
| `anima_character_96gb_dora.yaml` | anima_lora | DoRA + T-LoRA | 96GB | 角色 LoRA 顶配档:1024²、networkDim 32、batchSize 16 + gradAccum 1、attn=flash、cudagraph_trees、CMMD holdout 32。RTX PRO 6000 Blackwell / B200 / H100。峰值约 45-55 GB。 |
| `anima_style_96gb_loha.yaml` | anima_lora | LoHa + T-LoRA | 96GB | 画风 LoRA 顶配档,同硬件预算 BS 16 + LoHa r=16(等效 rank ~256)、宽 AR bucket、dropRate 0.18。峰值约 50-60 GB。 |

## 命名约定

```
<arch>_<purpose>_<vram-tier>[_<algo>].yaml
```

* `<arch>`:`anima` (Anima DiT) / `kohya` / `dp` (diffusion-pipe) — 当前都是 anima_lora 后端
* `<purpose>`:`lora`(通用)/ `character` / `style`
* `<vram-tier>`:`8gb` / `32gb` / 省略表示通用基线
* `<algo>`:`dora` / `loha`,与 default LoRA 的算法不同时显式标注

## 自定义起步

最快路径:

```bash
cp configs/anima_lora_default.yaml configs/my_recipe.yaml
# 改 dataset.source / output.name
lorahub validate configs/my_recipe.yaml
lorahub train configs/my_recipe.yaml
```

或者从 8GB / 32GB 模板出发,按硬件档位选起点。

## 字段含义

每份配方头部的注释块对每条非平凡设置都解释了:

* 为什么这个值(对应硬件预算 / 算法权衡)
* 跟其他档位比有什么差异
* 改动建议(VRAM 撑爆怎么办、想跑更快怎么办)

直接读 yaml 顶部即可。

## 常见调档

| 现象 | 调整 |
|------|------|
| 8GB 训练崩 OOM | `blocksToSwap` 24 → 28 / `networkDim` 8 → 4 |
| 16GB 卡想用 8gb 配置但留余量 | `blocksToSwap` 24 → 12 / `gradientCheckpointing` 留 true |
| 24GB 想用 32gb 配置 | `compileMode` 设为 null(或删该字段) / `validationSplitNum` 16 → 8 |
| 48GB 卡(RTX 6000 Ada / A6000)用 96gb 配置 | `batchSize` 16 → 8 / `attnMode` 改 torch(若没装 flash-attn) |
| 96GB 想榨满吞吐(可接受 LoRA 收敛 warning) | `batchSize` 16 → 32 / `numRepeats` × 0.5 / `learningRate` × √2 |
| 训练时间太长 | `numRepeats` 或 `epochs` 降一半 / 从 `default` 切到 `8gb`(epochs 已经更短) |

## 不再保留的旧配方

以下文件历史上存在过但因为 schema 变更 / 上游 API 不兼容删除了,如果你依赖它们请从 4 份当前配方迁移:

* `anima_character_24gb.yaml` / `anima_character_8gb.yaml` / `anima_character_8gb_dora.yaml`
* `anima_lora_starter.yaml` / `anima_lora_turbo.yaml`
* `anima_style_24gb.yaml` / `anima_style_8gb_loha.yaml`
* `sdxl_character_8gb.yaml` (kohya 后端,目前未维护)

迁移指南:character / style 8GB 用 `anima_lora_8gb.yaml` 起步, 24GB 用 `anima_lora_default.yaml` 然后按需打开 sampling/compile,32GB 直接用对应的 `anima_*_32gb_*.yaml`。
