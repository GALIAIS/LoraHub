---
title: CLI 命令
description: lorahub CLI 逐命令参考。
---

# CLI 命令

每条命令都接 `--help` 看完整 flag 表。下面给最常走的路径。

## `init`

在 `configs/` 起一份起步配置。

```powershell
lorahub init my_character
lorahub init my_style --template sdxl_style
lorahub init my_character --auto `
    --checkpoint C:\models\sdxl_base.safetensors `
    --dataset    .\datasets\my_character
```

`--auto` 会探测 `nvidia-smi` 显存,扫数据集图片数,从 checkpoint 文件名识别架
构(SDXL / Flux / SD3 / SD1.5,以及把 IllustriousXL / Pony / NoobAI /
Animagine 归为 SDXL),写一份 rank / batch / grad_accum 按显存档调好、
`numRepeats` 按数据集大小反比缩放的配置。`--vram-mib` 可手动覆盖检测。

## `bootstrap-kohya`

一键装 kohya-ss/sd-scripts:clone 仓库、建 venv、装 PyTorch + torchvision
(`--cuda cu121/cu124/cu128`,`--torch X.Y.Z`)、`pip install -r
requirements.txt`、装 xformers(`--no-xformers` 跳过)。`--force` 抹掉半装好的目
录。

```powershell
lorahub bootstrap-kohya                 # 默认: ./sd-scripts, cu124, torch 2.6.0
lorahub bootstrap-kohya --cuda cu121
lorahub bootstrap-kohya --no-xformers --force
```

## `bootstrap-diffusion-pipe`

一键装 tdrussell/diffusion-pipe:clone、建 venv、用 uv 装 PyTorch + DeepSpeed
+ 后端依赖。需要 dp 独占的架构(Flux2 / Wan / Cosmos / Z-Image 等)、或要在
共享架构(Anima)上对比 kohya 时用。

```powershell
lorahub bootstrap-diffusion-pipe
```

## `fetch-bangumi`

从 Hugging Face 上的 BangumiBase 数据集下载单角色图集,用来快速喂给冒烟测试。

```powershell
lorahub fetch-bangumi azurlaneanime                            # 列角色
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

每张图旁边落一个空 `.txt`,除非传 `--no-seed-captions`。

## `tag`

对图片目录跑 tagger,旁边写 kohya 风格 `.txt`。支持 WD14 / WD-v3
(ONNX,默认 `wd-eva02-large-tagger-v3`)和 JoyTag(PyTorch)。

```powershell
# 默认阈值: general=0.35 / character=0.85, 已有 caption 的跳过
lorahub tag ./datasets/akagi

# 全部重打、阈值收紧、递归
lorahub tag ./datasets/akagi --overwrite --general 0.45 -r

# JoyTag 后端(需 pip install lorahub[tagging])
lorahub tag ./datasets/akagi --tagger joytag --joytag-threshold 0.4
```

`--device auto` 在 `onnxruntime-gpu` 可用时挑 GPU;`--device cuda` 强制 GPU,
`--device cpu` 强制 CPU。

## `caption normalize`

批量跑 caption pipeline:atomic 变换、dropout 锚定、booru alias 重映射。
对配对的 `{image}.txt` 文件离线运行。

```powershell
lorahub caption normalize ./datasets/akagi --shuffle --keep-tokens 1 --booru-alias
```

## `anima-caption`

把已有 caption 重排成官方 Anima 格式:

```
masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
<1girl/solo/character>, @<trigger>,
<2-3 句自然语言描述>,
<其余 general tags>
```

```powershell
lorahub anima-caption ./datasets/akagi --trigger akagi --mode character
```

## `validate`

不 launch 训练,只校验配置。任何 `Severity.error` 都会让命令以非零退出。

```powershell
lorahub validate configs/my_character.yaml
```

## `info`

dry-run:加载配置、编译成 backend argv(kohya CLI flags 或 dp TOML),打印
entry script + 估算显存。**不动 GPU**。

```powershell
lorahub info configs/my_character.yaml
```

## `train`

跑完整训练。`Ctrl+C` 优雅停止 — runner 发 `CTRL_BREAK_EVENT`(Windows)或
`SIGINT`(Unix),子进程不退就 escalate 到 terminate / kill。
`KeyboardInterrupt`、`sigkill_handler` 等取消相关 traceback 会被 parser
识别,不当作错误渲染。

```powershell
lorahub train configs/my_character.yaml
lorahub train configs/my_character.yaml --workspace .\runs\my_character_v1
```

Job 产物(logs、checkpoints、samples、`events.jsonl`)默认落
`runs/<output.name>/`。step 节奏 checkpoint(`saveEveryNSteps`)与 epoch 节奏
checkpoint(`saveEveryNEpochs`)互斥 — 两者同时配且 step 设置生效时,epoch
flag 会被丢掉,避免对齐边界上的双写。

## `sweep`

规划或跑一次超参 sweep。当前 expander 是网格 + 随机 + TPE(三种 mode);
ASHA 已评估为不可接,见 audit。

```powershell
lorahub sweep configs/my_character.yaml --axis lr=1e-4,5e-5 --axis rank=16,32 --dry-run
lorahub sweep configs/my_character.yaml --axis lr=1e-4,5e-5
```

`--dry-run` 枚举 variant 不启动 job。否则每个 variant 写成独立 config 提交到
`POST /api/sweeps`。

## `serve`

启动 LoraHub FastAPI 服务器。需要 `api` extras(`pip install lorahub[api]`)。

```powershell
lorahub serve --port 18765
lorahub serve --host 0.0.0.0 --port 8000 --reload
```

默认绑 `127.0.0.1`,无认证 — 仅适用于本机。

## `version`

打印 `lorahub` 版本。

```powershell
lorahub version
```

## `jobs ls`

列训练任务。默认走本机 API(`http://127.0.0.1:18765`),`--api` 可换。

```powershell
lorahub jobs ls
lorahub jobs ls --status running
lorahub jobs ls --limit 20 --json
```

## `jobs show`

按 id 看单 job 详情(状态、运行时长、步数、最近事件、产物路径)。

```powershell
lorahub jobs show 0192a3f4-...
```

## `jobs cancel`

向运行中的 job 发取消信号(等价 `DELETE /api/jobs/{id}`)。子进程优雅退出。

```powershell
lorahub jobs cancel 0192a3f4-...
```

## `jobs kill`

`SIGKILL` 进程组(等价 `POST /api/jobs/{id}/kill`)。仅在 trainer 卡死、
`cancel` 不响应时用。

```powershell
lorahub jobs kill 0192a3f4-...
```

## `jobs resume`

从最近一次 DeepSpeed checkpoint 续训(等价 `POST /api/jobs/{id}/resume`)。
要求 `backend.diffusionPipe.checkpointEveryNMinutes` 或
`checkpointEveryNEpochs` 在原 job 已开。

```powershell
lorahub jobs resume 0192a3f4-...
```

## `jobs rerun`

原地重跑同一 job(等价 `POST /api/jobs/{id}/rerun`)。`events.jsonl` 会被清,
是 plain rerun 而非 resume。

```powershell
lorahub jobs rerun 0192a3f4-...
```

## `sweeps submit`

把一份 sweep 计划提交到 API。支持 `--mode {grid,random,tpe}`、
`--n-trials`、`--seed`。TPE 走 ask-tell 流式反馈,跨重启自动续投。

```powershell
lorahub sweeps submit configs/my_character.yaml `
    --axis lr=1e-4,5e-5 --axis rank=16,32 `
    --mode tpe --n-trials 20 --seed 42
```

## `sweeps ls`

列已提交的 sweeps,带状态、trial 数、最新 score。

```powershell
lorahub sweeps ls
lorahub sweeps ls --status running --json
```

## `system gpu`

一次性 GPU 快照(等价 `GET /api/system/stats` 的 GPU 切片):每张卡的
显存占用、温度、利用率、当前进程。

```powershell
lorahub system gpu
lorahub system gpu --json
```

## `system info`

host CPU / RAM / 磁盘 / 网络快照(等价 `GET /api/system/stats`)。在远程
部署上不开浏览器看面板时方便。

```powershell
lorahub system info
```
