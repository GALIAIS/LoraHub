---
title: 端到端冒烟测试
description: 从图片到 caption 到训练再到 preview 的全链路冒烟测试。Run the full data → caption → train → preview pipeline against a real character set.
---

# 端到端冒烟测试

后端装好(已设 `LORAHUB_KOHYA_SD_SCRIPTS` / `LORAHUB_DIFFUSION_PIPE` 或
复制 `.env.example`)、基础模型在位之后,从零到训练好的 LoRA + live
preview 长这样:

```powershell
# 1. 从 BangumiBase 拉一个角色的图
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. 给每张图打 caption,二选一:

#    a) 经典 auto-tag(WD14 / WD-v3 ONNX, 最快):
lorahub tag ./datasets/laffey

#    b) 智能 caption(WD14 + 视觉 LLM, Anima 格式):
curl -X POST http://127.0.0.1:18765/api/image-studio/ai/smart-caption \
     -H 'Content-Type: application/json' \
     -d '{"path":"./datasets/laffey","captionMode":"character","triggerWord":"laffey"}'

# 3. 起一份配置并编辑(把 baseModel.checkpoint 指到你的模型)
lorahub init smoke
notepad configs/smoke.yaml

# 4. 干跑校验
lorahub validate configs/smoke.yaml
lorahub info     configs/smoke.yaml

# 5. 训练
lorahub train    configs/smoke.yaml
```

!!! success "参考耗时"
    在 RTX 4070 Laptop(8 GB 显存)、IllustriousXL 作底模的条件下,这条
    路径在 3 分钟内出一份 21 MB 的 SDXL LoRA 文件 — 用 3 张 BangumiBase
    "laffey (azur lane)" 图,512×512,2 步实测过。

## 要快速测数据?

`lorahub fetch-bangumi` 从 [BangumiBase](https://huggingface.co/BangumiBase)
拉单个角色的图集 — 上游已聚类、MIT license,正适合冒烟。

```powershell
# 列某部作品的角色
lorahub fetch-bangumi azurlaneanime

# 拉角色 5 的最多 50 张图
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50

# 或先拉 8 张缩略图先认人
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

每张图旁边落一个空的 `.txt` caption 文件 — 训练前要么手填,要么用接下来
的步骤自动打标。

## 用 WD14 / JoyTag 自动打标

`lorahub tag` 跑 tagger 扫一个目录,旁边写 kohya 风格的 `.txt` caption。

```powershell
# 默认阈值(general=0.35, character=0.85),已有 caption 的图跳过
lorahub tag ./datasets/akagi

# 全部重新打标,general 阈值收紧
lorahub tag ./datasets/akagi --overwrite --general 0.45

# 训风格 / 概念 LoRA 时跳过 character tag
lorahub tag ./datasets/akagi --no-include-character

# JoyTag 后端(PyTorch, ~5800 标签词表, 默认 0.4 阈值)
lorahub tag ./datasets/akagi --tagger joytag --joytag-threshold 0.4
```

WD14 默认模型是 `SmilingWolf/wd-eva02-large-tagger-v3`。CPU 推理几百张图
按 ~1 s/张算够用;批量吞吐想再快就装 GPU 运行时:

```powershell
pip uninstall onnxruntime
pip install lorahub[gpu]              # 或: pip install onnxruntime-gpu
lorahub tag ./datasets/akagi --device cuda
```

`--device auto` 在 `onnxruntime-gpu` + CUDA 12.x 都到位时挑 GPU,否则
回退 CPU。`--device cuda` 强制 GPU,缺了会报 actionable 错误。

## 智能 caption(WD14 + 视觉 LLM)

Image Studio 的 smart-caption 把 WD14 与配置好的视觉 LLM 串起来,产出
Anima 格式的 caption:

```
masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
<1girl/solo/character>, @<trigger>,
<2-3 句自然语言描述>,
<其余 general tags>
```

三种模式:

- **style** — 显式描述介质和渲染,把风格绑到 trigger word 上。
- **character** — 跳过固定身份特征(发色 / 瞳色 / 标志服装),让模型从
  latent 中自己学出来。
- **general** — 全描述;数据集不靠 trigger 时用。

> smart-caption 现在以后台 session 形式运行:`POST /api/image-studio/ai/smart-caption`
> 同步返回 `202 + session_id`,前端走 `GET .../status/{id}` 轮询,需要中
> 断走 `POST .../cancel/{id}`。

每行 caption 后续可由 live preview worker 重新渲成预览图(走
diffusion-pipe 训练时,见下一节)。

## 训练时的 live preview

走 diffusion-pipe 训练时,LoraHub 拉起一个后台 worker 监视
`runs/<job>/output/{step|epoch}{N}/`,每出一个 checkpoint 就按 prompt
渲一张 PNG。默认 prompt 文件在 `configs/sample_prompts/anima_default.txt`,
里面换 trigger 即可换主体。

worker 对 `checkpoint_saved` 事件 < 1s 响应,5 s 轮询作兜底。每 checkpoint
渲染预算压制使训练吞吐保持在 baseline 的 ~70%。被跳过的渲染(显存不够、
取消)会静默重排,只有真正崩溃才发 error 事件。

PNG 落到 `workspace/samples/step{N}_{idx}.png`,Jobs 页的 **Sample
Gallery** 实时显示。

---

## English

Once you have a backend installed and a base model on disk, the
end-to-end smoke test is the same five steps shown above:
`fetch-bangumi` → `tag` (or `smart-caption`) → `init` → `validate` +
`info` → `train`. On an RTX 4070 Laptop (8 GB) with IllustriousXL as
the base, three BangumiBase images at 512×512 and 2 steps produce a
21 MB SDXL LoRA in under three minutes.

`fetch-bangumi` pulls character-clustered datasets from
[BangumiBase](https://huggingface.co/BangumiBase) (MIT-licensed). Each
image is dropped next to an empty `.txt` caption file unless you pass
`--no-seed-captions`.

`lorahub tag` writes kohya-style `.txt` captions next to images.
WD14 defaults to `SmilingWolf/wd-eva02-large-tagger-v3` and skips
files that already have a non-empty caption. CPU is fine for a few
hundred images at ~1 s each; install `lorahub[gpu]` and pass
`--device cuda` for batch throughput.

The Image Studio's smart-caption pipeline composes WD14 with a vision
LLM and produces an Anima-format caption (`masterpiece, best quality,
score_7, <rating>, <subject>, @<trigger>, <2-3 sentences>, <general
tags>`). It now runs as a background session: the POST returns
`202 + session_id`, the client polls `GET .../status/{id}` for
progress, and `POST .../cancel/{id}` aborts it.

When training with diffusion-pipe, LoraHub spins up a background live
preview worker that watches `runs/<job>/output/{step|epoch}{N}/` and
renders one PNG per prompt for every new checkpoint. Default prompts
live at `configs/sample_prompts/anima_default.txt`. PNGs land in
`workspace/samples/step{N}_{idx}.png` and surface in the Sample
Gallery.