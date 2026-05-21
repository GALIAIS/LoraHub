# 8GB 笔记本 anima_lora 训练手册

两份配置 — 人物 / 风格各一份,针对 RTX 3060/3070/4060/4070 mobile (8 GB VRAM, 16-32 GB 系统内存):

| 文件 | 算法 | 用途 | 预期吞吐 | 预期 VRAM 峰值 |
|---|---|---|---|---|
| `anima_character_8gb_dora.yaml` | DoRA | 人物 / 角色身份 | ~6-12 s/step | ~7.0-7.5 GB |
| `anima_style_8gb_loha.yaml` | LoHA | 风格 / 笔触 / 色调 | ~7-12 s/step | ~7.0-7.5 GB |

> ⚠ 这是 anima_lora 后端配置(`backend.type: anima_lora`)。如果要 diffusion-pipe,看 `anima_character_8gb.yaml`。

## 8 GB 怎么塞进去 — 显存预算

| 项 | 默认 (24 GB) | 8 GB 配置 | 来源 |
|---|---|---|---|
| DiT 权重 (28 blocks) | ~7.0 GB | ~1.0 GB | `blocks_to_swap: 24` 把 24/28 块推到 CPU |
| Activation (1024² → 768²) | ~800 MB | ~440 MB | `resolution: [768, 768]`,token 4096 → 2304 |
| Activation checkpoint | resident | offload to CPU | `unsloth_offload_checkpointing` + `cpu_offload_checkpointing` |
| Optimizer state | ~2x 参数量 (fp32) | ~0.5x | `optimizerType: AdamW8bit` (bitsandbytes) |
| LoRA 参数 | <100 MB | <100 MB | `networkDim: 8` |
| VAE forward 峰值 | ~1.5 GB | ~150 MB | `vaeChunkSize: 1` |
| Sample 生成峰值 | +1-2 GB | 关闭 | `sampling.enabled: false` |
| **VRAM 峰值** | ~14 GB | **~7.0 GB** | |

留 0.5-1 GB 给桌面/浏览器。

## 算法为什么这么挑

### 人物 → DoRA

```
W' = m · (W₀ + ΔW) / ‖W₀ + ΔW‖_c
ΔW = α/r · B·A   (LoRA legs)
m  = per-output-channel magnitude vector  (DoRA 新增)
```

DoRA 把权重拆成"方向"和"幅度"。LoRA 的 `B·A` 学方向,而 magnitude 向量 `m` 学幅度尺度。在小数据集 (≤50 张图) 的人物训练里,身份特征往往体现为某些通道的特定权重幅度;DoRA 显式建模这个,在相同参数量下识别度更高。

存储与原版 LoRA 一致 (ComfyUI 原生兼容,加载文件时自动识别 `.dora_scale`)。

### 风格 → LoHA

```
ΔW = (W₁ₐ · W₁ᵦ) ⊙ (W₂ₐ · W₂ᵦ)
```

两个 LoRA 对的 Hadamard 积。有效秩可达 `r²` (经典 LoRA 是 `r`),参数量是 LoRA 的 2 倍。

风格特征是多尺度耦合 — 笔触受色调影响,色调受光照影响。纯 LoRA 的低秩 `B·A` 没法表达这种跨秩耦合,而 Hadamard 积刚好。

## 启动训练

### 1. 准备数据集

每张图 + 同名 `.txt` caption 文件。建议:
- **人物**: 30-80 张图,每张 caption 强调身份 tags (1girl, hair colour, eye colour, outfit pieces)。第一个 token 通常是触发词 (如 `@your_character`),启动 `keepTokens: 1` 锁住。
- **风格**: 50-300 张图,caption 描述场景/构图/对象,**不带**身份 tags (避免"风格 = 这个角色"的串味)。

```bash
# 用内置 smart-caption 生成一致的 tag 集
lorahub captions smart ./datasets/your_character --trigger @your_character --mode character
lorahub captions smart ./datasets/your_style --mode style
```

### 2. 编辑配置

打开对应 yaml,修改 3 处:

```yaml
baseModel:
  checkpoint: ./models/diffusion_models/anima-base-v1.0.safetensors  # ← 你的 anima ckpt 路径
dataset:
  source: ./datasets/your_character          # ← 你的数据集目录
output:
  name: my_character_v1                      # ← 你想要的输出文件名
```

### 3. 提交训练

```bash
lorahub jobs submit configs/anima_character_8gb_dora.yaml
# 或
lorahub jobs submit configs/anima_style_8gb_loha.yaml
```

或在 Web UI:
1. 打开 http://127.0.0.1:6006
2. Configs → Import → 选 yaml
3. Run

## 故障排查

| 现象 | 原因 | 修复 |
|---|---|---|
| OOM during forward | block swap 不够 | `blocksToSwap: 26` (anima 上限) |
| OOM during VAE cache | VAE batch 太大 | `vaeChunkSize: 1` (已是默认) |
| OOM during sample | sample 暂时驻留 | `sampling.enabled: false` (已是默认) |
| step 1 慢但稳 | 第一次缓存 TE/LLM/VAE 输出 | 正常,第 1 epoch 后会快 |
| step >15s | laptop 进电池模式 | 接电源 / 性能模式 / 关闭 GPU power saver |
| loss diverge / NaN | lr 偏高 | DoRA: 3e-5 / LoHA: 5e-5 |
| 触发词不工作 | caption dropout 太高 | 人物 `dropRate: 0.0` + `keepTokens: 1` |
| 风格不像 | repeats 太少 / lr 太低 | `numRepeats: 5` 或 `learningRate: 1e-4` |
| 训练中崩溃 | 别的进程争 VRAM | 关 ComfyUI / Stable Diffusion WebUI / 浏览器硬件加速 |

`nan_guard` 已开 — 偶尔的 NaN 尖峰会自动半减 lr 并从 EMA 影子恢复 (训练不会停)。

## 调优空间

### 想训得更快 (牺牲质量)

```yaml
schedule:
  epochs: 4         # ↓ 6 → 4
dataset:
  numRepeats: 4     # ↓ 6 → 4
backend:
  animaLora:
    blocksToSwap: 20    # ↓ 24 → 20  仅当 OOM 没出现时
```

### 想训得更好 (牺牲速度)

```yaml
schedule:
  epochs: 12        # ↑
backend:
  animaLora:
    networkDim: 16  # ↑ 8 → 16  显存边缘,可能 OOM
    learningRate: 3.0e-05  # ↓
```

### 切换算法对比

如果有时间 / 想看效果差异,可以同 dataset 跑一次 `algorithm: lora`,一次 `dora`,比较哪个更像。换 `algorithm:` 一行即可,其他全部不变。

## 估算总时长

`总 step ≈ (numRepeats × imgs × epochs) / (batchSize × gradAccum)`

例子 (人物配置):
- 60 imgs × 6 repeats × 6 epochs / (1 × 8) = **270 steps**
- 270 × 8s/step ≈ **36 分钟**

例子 (风格配置):
- 150 imgs × 4 repeats × 8 epochs / (1 × 8) = **600 steps**
- 600 × 9s/step ≈ **90 分钟**
