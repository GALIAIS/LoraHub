---
title: Anima Base v1.0 模型架构
description: 基于本地 safetensors 权重、LoraHub anima_lora 实现和公开模型卡的 Anima Base v1.0 技术说明。
---

# Anima Base v1.0 模型架构

本文档记录 LoraHub 当前接入的 Anima Base v1.0 的可核验事实、权重结构、推理/训练数据流和相关论文谱系。结论来自三类证据：

- 本地权重：`E:\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors`
- 当前代码：`external/anima_lora/library/anima/` 与 `external/anima_lora/library/models/qwen_vae.py`
- 公开模型卡：[`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)

需要注意：本地 safetensors 没有嵌入 metadata，不能仅从权重文件恢复训练配方、数据清单或完整论文声明。本文把权重/代码实测信息和公开资料分开标注。

## 公开模型信息

Anima 是 CircleStone Labs 与 Comfy Org 合作发布的约 20 亿参数 text-to-image 模型。公开模型卡说明它主要面向 anime 概念、角色、风格和非写实艺术图像，不以照片级写实为目标。

公开模型卡给出的关键事实：

| 项 | 信息 |
| --- | --- |
| 版本 | `Anima-Base`，预训练未精修 base model |
| 基座 | `nvidia/Cosmos-Predict2-2B-Text2Image` 派生模型 |
| 模型定位 | 插画、anime、艺术图像生成 |
| 数据 | 数百万 anime 图像，约 80 万非 anime 艺术图像，未使用 synthetic data |
| 数据知识截止 | anime 训练数据截止到 2025-09 |
| 推荐分辨率 | 512^2 到 1536^2 |
| 推荐采样 | 30-50 steps，CFG 4-5 |
| 相关文件 | DiT 权重、Qwen3 0.6B 文本编码器、Qwen-Image VAE |
| 许可证 | CircleStone Labs Non-Commercial License，同时作为 Cosmos-Predict2 派生模型受 NVIDIA Open Model License 约束 |

模型卡还明确给出微调建议：不要训练 LLM Adapter，rank 32 LoRA 可从 `2e-5` 左右低学习率起步。这一点和本项目的默认 Anima 微调策略一致：LLM Adapter 对文本语义进入 DiT 前的表示有很强影响，训练它很容易破坏已有语义对齐。

## 本地权重检查

检查命令使用 `safetensors.safe_open(..., framework="numpy")` 只读取 tensor shape/dtype，没有把 BF16 tensor 全量加载进内存。

本地文件摘要：

| 项 | 值 |
| --- | --- |
| 路径 | `E:\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors` |
| 文件大小 | 4,182,218,328 bytes |
| safetensors metadata | `None` |
| tensor 数 | 685 |
| dtype | 全部 BF16 |
| 参数量 | 2,091,068,928 |
| BF16 权重体积 | 约 4.182 GB |

按模块参数分布：

| 前缀 | 参数量 | 占比 |
| --- | ---: | ---: |
| `net.blocks` | 1,937,782,784 | 92.67% |
| `net.llm_adapter` | 134,663,680 | 6.44% |
| `net.t_embedder` | 16,777,216 | 0.80% |
| `net.final_layer` | 1,703,936 | 0.08% |
| `net.x_embedder` | 139,264 | 0.007% |
| `net.t_embedding_norm` | 2,048 | 0.0001% |

权重 key 全部位于 `net.*` 命名空间。主干 DiT block 编号为 `0..27`，共 28 层；LLM Adapter block 编号为 `0..5`，共 6 层。

## DiT 主干结构

当前 `load_dit_model()` 按固定配置实例化 Anima：

```python
max_img_h=512
max_img_w=512
max_frames=128
in_channels=16
out_channels=16
patch_spatial=2
patch_temporal=1
model_channels=2048
num_blocks=28
num_heads=16
mlp_ratio=4.0
crossattn_emb_channels=1024
use_adaln_lora=True
adaln_lora_dim=256
use_llm_adapter=True
```

结构表：

| 部件 | 实测/代码值 |
| --- | --- |
| VAE latent channels | 16 |
| patch | temporal 1, spatial 2x2 |
| patch 输入维度 | 68 |
| hidden size | 2048 |
| DiT blocks | 28 |
| attention heads | 16 |
| head dim | 128 |
| MLP hidden | 8192 |
| cross attention context dim | 1024 |
| AdaLN-LoRA dim | 256 |
| final patch 输出 | 64 |

`net.x_embedder.proj.1.weight` shape 是 `[2048, 68]`。68 的来源是：

- latent：`16 channels * 1 temporal patch * 2 * 2 = 64`
- padding mask：`1 channel * 1 * 2 * 2 = 4`
- 合计：68

每个 DiT block 有 20 个 tensor，单层约 69,206,528 参数：

| 子模块 | tensor shape |
| --- | --- |
| self-attn q/k/v/o | `[2048, 2048]` |
| self-attn q_norm/k_norm | `[128]` |
| cross-attn q/o | `[2048, 2048]` |
| cross-attn k/v | `[2048, 1024]` |
| cross-attn q_norm/k_norm | `[128]` |
| MLP layer1 | `[8192, 2048]` |
| MLP layer2 | `[2048, 8192]` |
| AdaLN down projection | `[256, 2048]`，self/cross/mlp 各一组 |
| AdaLN up projection | `[6144, 256]`，self/cross/mlp 各一组 |

每个 block 的计算顺序是：

1. 用 timestep embedding 生成 AdaLN 的 shift/scale/gate。
2. 对 latent tokens 做 self-attention，带 3D RoPE。
3. 对文本上下文做 cross-attention。
4. 通过 4x hidden 的 MLP。
5. 每个子层用 gate 控制残差注入。

这里的 AdaLN-LoRA 不是外部 LoRA 微调层，而是模型内部的低秩调制结构：先把 timestep embedding 从 2048 降到 3 组 256，再分别升到 self-attn、cross-attn、MLP 需要的 `3 * 2048` 调制参数。

## LLM Adapter

Anima 使用 Qwen3/Qwen3.5 文本编码器生成 source hidden states，再由 LLM Adapter 转成 DiT 使用的 1024 维 cross-attention 上下文。

LLM Adapter 实测结构：

| 项 | 值 |
| --- | --- |
| blocks | 6 |
| model dim | 1024 |
| heads | 16 |
| head dim | 64 |
| vocabulary | 32128 |
| embed weight | `[32128, 1024]` |
| output projection | `[1024, 1024]` |

每个 Adapter block 包含 self-attention、cross-attention、MLP 和 RMSNorm。Adapter attention 使用 QK norm 和 RoPE。它的目标不是生成图像 token，而是把 Qwen 文本语义桥接到类似 T5 目标 token 空间，再交给 DiT cross-attention。

微调含义：

- LLM Adapter 参数约 1.35 亿，只占模型 6.44%，但它位于文本语义进入 DiT 之前。
- 它对 prompt 对齐影响大，少量数据 LoRA/风格训练时训练 Adapter 容易造成语义漂移。
- 默认应冻结 Adapter；只有大规模、有明确文本对齐目标的数据才考虑单独低学习率训练。

## VAE 与 latent 空间

Anima 使用 Qwen-Image VAE。当前实现位于 `external/anima_lora/library/models/qwen_vae.py`，类名是 `AutoencoderKLQwenImage`。

代码中的 VAE 默认结构：

| 项 | 值 |
| --- | --- |
| base dim | 96 |
| latent dim | 16 |
| dim mult | `[1, 2, 4, 4]` |
| residual blocks | 2 |
| temporal downsample | `[False, True, True]` |
| spatial compression | `2 ** 3 = 8` |
| tiling minimum | 256x256 |
| tiling stride | 192x192 |

所以单张图片在 DiT 侧的 latent shape 是：

```text
B, C, T, H/8, W/8
C = 16
T = 1
```

例如 1024x1024 图片进入 DiT 前是 `1 x 16 x 1 x 128 x 128`，patch 2x2 后是 `4096` 个 token。用户之前使用的 912x1632 会对应 latent 约 `114 x 204`，patch 后约 `57 x 102 = 5814` token，显著高于 1024x1024 的 4096 token，attention 显存和数值压力都会上升。

## 训练和采样目标

当前 Anima 训练实现是 rectified flow / flow matching 风格。代码中采样器注释为 Euler discrete for rectified flow，sigma 从 1.0 线性走到 0.0，并可用 `flow_shift` 重映射：

```text
sigma_i = linspace(1, 0)
if flow_shift != 1:
  sigma = sigma * flow_shift / (1 + (flow_shift - 1) * sigma)
```

采样过程从纯噪声 `x` 开始，DiT 预测 vector field，然后 Euler 更新：

```text
x = x + model_output * (sigma_next - sigma)
```

训练侧的 rectified flow 路径可以概括为：

```text
z_sigma = (1 - sigma) * latent + sigma * noise
target  = noise - latent
```

loss 是对预测 vector field 和 target 的 MSE，并可叠加 timestep weighting，例如 `none`、`cosmap`、`sigma_sqrt`、`min_snr_rf`。

## 推理数据流

完整推理路径：

1. Prompt 由 Qwen3/Qwen3.5 tokenizer 和 text encoder 编码。
2. 如果启用 LLM Adapter，Qwen hidden states 进入 6 层 Adapter，得到 1024 维 cross-attention context。
3. 随机噪声 latent 以 `B x 16 x 1 x H/8 x W/8` 进入 DiT。
4. padding mask 与 latent 拼接后 patchify，形成 token 序列。
5. timestep embedding 生成 AdaLN 调制信号。
6. 28 层 DiT block 反复执行 self-attention、cross-attention、MLP。
7. final layer 输出 patch latent，再 unpatchify 回 `16 x H/8 x W/8`。
8. Euler rectified-flow 更新 latent。
9. 最终 latent 经 Qwen-Image VAE decode 成图像。

## 与相关论文/模型的关系

Anima 没有在本地 checkpoint metadata 中给出独立论文信息。它更准确的定位是：公开模型卡声明的 Cosmos-Predict2 派生模型，加上当前代码中可见的 DiT、Flow Matching、RoPE、AdaLN 和 Qwen-Image VAE 组合。

相关技术谱系：

| 方向 | 关联 |
| --- | --- |
| DiT | Anima 用 transformer 替代 U-Net 主干，在 latent patch 上建模。对应 DiT 论文的核心思想：latent patch token + transformer scaling。 |
| Flow Matching | 当前训练/采样是 rectified flow 风格，模型预测 noise-to-data vector field。 |
| RoPE | 主干使用 3D RoPE，LLM Adapter 使用文本 RoPE，用于序列/空间位置注入。 |
| AdaLN / AdaLN-Zero | DiT block 使用 timestep-conditioned AdaLN 调制 shift/scale/gate。 |
| Qwen-Image | VAE 来自 Qwen-Image 系，latent channel 为 16，空间压缩比为 8。 |
| Qwen3 | 文本编码器使用 Qwen3/Qwen3.5，Adapter 把 Qwen hidden states 转为 DiT cross-attention context。 |
| Cosmos-Predict2 | 公开模型卡声明 Anima 是 `nvidia/Cosmos-Predict2-2B-Text2Image` 的派生模型，当前代码 docstring 也称其为 Cosmos-Predict2 DiT model。 |

可直接引用的公开资料：

- Anima model card: https://huggingface.co/circlestone-labs/Anima
- DiT: https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747
- RoPE / RoFormer: https://arxiv.org/abs/2104.09864
- Qwen-Image: https://huggingface.co/Qwen/Qwen-Image and https://arxiv.org/abs/2508.02324

## 对 LoraHub 训练的实际影响

Anima Base 是一个 true base model，不是强审美精修模型。训练 LoRA 时如果效果差，优先检查数据、caption、学习率和 Adapter 冻结，而不是先提高训练强度。

建议：

- 风格 LoRA：低学习率，rank 16-32 起步；公开模型卡建议 rank 32 从 `2e-5` 左右试。
- 小数据集：冻结 LLM Adapter，避免文本语义桥接层被少量样本拉偏。
- V100：BF16 不可用，纯 FP16 在高 token 数 DiT block stack 中数值风险高；应使用 AMP + FP32 islands，或直接 FP32 稳定训练。
- 高分辨率：token 数按 latent patch 网格增长，912x1632 这类长边构图比 1024x1024 更吃 attention 显存和数值稳定性。
- prompt：模型卡推荐 tag 与自然语言混用；artist tag 需要 `@` 前缀；质量、安全、年份、meta tags 的顺序会影响稳定性。
- 采样：默认可以先用 30-50 steps、CFG 4-5；采样器可从 `er_sde`、`euler_a`、`dpmpp_2m_sde_gpu` 对比。

## LoRA 注入优先级

按权重结构和训练风险，推荐的 LoRA 注入优先级：

1. DiT attention projection：`self_attn.q/k/v/output_proj` 与 `cross_attn.q/k/v/output_proj`。
2. DiT MLP：表达力更强，但参数和过拟合风险更高。
3. AdaLN modulation：会改变 timestep 条件调制，适合明确风格控制实验，不建议默认训练。
4. LLM Adapter：默认冻结；除非目标就是重新学习文本桥接，否则不应动。
5. VAE：不属于 Anima DiT LoRA 常规目标，不建议跟随风格/角色 LoRA 一起训练。

## 检查命令

本地权重结构可用下面的只读命令复查：

```powershell
.\.venv\Scripts\python.exe -c "from safetensors import safe_open; from math import prod; from collections import Counter; p=r'E:\ComfyUI\models\diffusion_models\anima-base-v1.0.safetensors'; f=safe_open(p, framework='numpy'); total=0; by=Counter(); d=Counter(); keys=list(f.keys()); 
for k in keys:
    s=f.get_slice(k); n=prod(s.get_shape()); total+=n; by['.'.join(k.split('.')[:2])]+=n; d[s.get_dtype()]+=n
print(len(keys), total, dict(d)); print(by)"
```

如果后续替换 checkpoint，应重新跑这类检查。只要 tensor 数、block 数、hidden size 或 Adapter shape 改变，本文档中的架构结论就不再自动适用。
