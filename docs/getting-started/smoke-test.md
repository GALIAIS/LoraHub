---
title: 端到端教程
description: 从原始图片到训练好的 LoRA 全流程示例。
---

# 端到端教程

后端装好（已设 `LORAHUB_KOHYA_SD_SCRIPTS` / `LORAHUB_DIFFUSION_PIPE`，或复制 `.env.example` 后编辑）、基础模型在位之后，从零到训练好的 LoRA + 实时样本图，全链路示意如下。

```powershell
# 1. 从 BangumiBase 拉一个角色的图
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. 给每张图打 caption（任选一种方式）

#    a) 经典自动打标（WD14 / WD-v3 ONNX，最快）
lorahub tag ./datasets/laffey

#    b) 智能打标（WD14 + 视觉 LLM，Anima 格式）
curl -X POST http://127.0.0.1:18765/api/image-studio/ai/smart-caption \
     -H 'Content-Type: application/json' \
     -d '{"path":"./datasets/laffey","captionMode":"character","triggerWord":"laffey"}'

# 3. 起一份配置并编辑（把 baseModel.checkpoint 指到你的模型）
lorahub init smoke
notepad configs/smoke.yaml

# 4. 干跑校验
lorahub validate configs/smoke.yaml
lorahub info     configs/smoke.yaml

# 5. 训练
lorahub train    configs/smoke.yaml
```

!!! success "参考耗时"
    在 RTX 4070 Laptop（8 GB 显存）、IllustriousXL 作底模的条件下，这条路径在 3 分钟内出一份 21 MB 的 SDXL LoRA 文件 —— 用 3 张 BangumiBase「laffey (azur lane)」图，512×512，2 步实测过。

## 准备数据集

`lorahub fetch-bangumi` 从 [BangumiBase](https://huggingface.co/BangumiBase) 拉单个角色的图集。上游已聚类、MIT license，正适合冒烟测试。

```powershell
# 列某部作品的角色
lorahub fetch-bangumi azurlaneanime

# 拉角色 5 的最多 50 张图
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50

# 或先拉 8 张缩略图先认人
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

每张图旁边落一个空的 `.txt` caption 文件 —— 训练前要么手填，要么用接下来的步骤自动打标。

## 用 WD14 / JoyTag 自动打标

`lorahub tag` 跑 tagger 扫一个目录，旁边写 kohya 风格的 `.txt` caption。

```powershell
# 默认阈值（general=0.35，character=0.85），已有 caption 的图跳过
lorahub tag ./datasets/akagi

# 全部重新打标，general 阈值收紧
lorahub tag ./datasets/akagi --overwrite --general 0.45

# 训风格 / 概念 LoRA 时跳过 character 标签
lorahub tag ./datasets/akagi --no-include-character

# 国内网络无法访问 Hugging Face 时使用已验证的 ModelScope 镜像
lorahub tag ./datasets/akagi --source modelscope

# JoyTag 后端（PyTorch，~5800 标签词表，默认 0.4 阈值）
lorahub tag ./datasets/akagi --tagger joytag --joytag-threshold 0.4
```

WD14 默认模型是 `SmilingWolf/wd-eva02-large-tagger-v3`。CPU 推理几百张图按约 1 秒每张算够用；批量吞吐想再快就装 GPU 运行时：

`--source auto`（默认）会读取「设置 → 网络加速 → 优先 ModelScope」。启用后，
内置 WD14 模型先从 `fireicewolf/*` 魔搭镜像下载，失败再回退 Hugging Face；
命令行可用 `--source modelscope` 固定走魔搭。模型保存到项目内
`models/modelscope/hub/`，中断后会从 `.lorahub.part` 断点续传。

```powershell
pip uninstall onnxruntime
pip install lorahub[gpu]              # 或：pip install onnxruntime-gpu
lorahub tag ./datasets/akagi --device cuda
```

`--device auto` 在 `onnxruntime-gpu` + CUDA 12.x 都到位时挑 GPU，否则回退 CPU。`--device cuda` 强制 GPU，缺了会报清晰的错误。

## 智能打标（WD14 + 视觉 LLM）

Image Studio 的 smart-caption 把 WD14 与配置好的视觉 LLM 串起来，产出 Anima 格式的 caption：

```
masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
<1girl|solo|character>, @<trigger>,
<2-3 句自然语言描述>,
<其余 general tags>
```

三种模式：

- **style** — 显式描述介质和渲染，把风格绑到 trigger word 上。
- **character** — 跳过固定身份特征（发色 / 瞳色 / 标志服装），让模型从 latent 中自己学出来。
- **general** — 全描述；数据集不靠 trigger 时用。

smart-caption 以后台 session 形式运行：`POST /api/image-studio/ai/smart-caption` 同步返回 `202 + session_id`，前端走 `GET .../status/{id}` 轮询，需要中断走 `POST .../cancel/{id}`。

## 训练中的实时样本图

不同后端出样本图的路径不同，但落盘后都会被 Web UI 的 Sample Gallery 自动收到。

- **kohya** — train.py 内置 sampler，按 `cfg.sampling.everyNEpochs` / `everyNSteps` 出图，PNG 落到 `runs/<job>/ckpt/sample/`。
- **diffusion-pipe** — 上游不出图。LoraHub 拉起一个后台 PreviewWorker 监视 `runs/<job>/output/{step|epoch}{N}/`，每出一个 checkpoint 就按 prompt 渲一张 PNG。开关在 `cfg.sampling.enableLiveInference`。
- **anima_lora** — 复用上游 train.py 的 sampler。`cfg.sampling.enabled=true` 即可，未指定 prompts 文件时 LoraHub 自动用数据集前 3 条 caption 生成默认 prompts。PNG 落到 `runs/<job>/ckpt/sample/`。

PreviewWorker（diffusion-pipe）对 `checkpoint_saved` 事件 < 1 秒响应，5 秒轮询作兜底。每个 checkpoint 的渲染预算压制使训练吞吐保持在 baseline 的约 70%。被跳过的渲染（显存不够、取消）会静默重排，只有真正崩溃才发 error 事件。
