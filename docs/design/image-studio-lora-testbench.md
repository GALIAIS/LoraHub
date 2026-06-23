# 图像工作台 LoRA 生图测试页设计

目标是在图像工作台新增一个“LoRA 测试”页面，用训练出的 LoRA 快速做生图验收：选模型、调参数、批量生成、对比结果、保存可复现实验。它不是训练时 checkpoint 预览的替代品，而是训练完成后由用户主动测试 LoRA 效果的工作台。

本文只做设计，不实现代码。

## 设计原则

- 页面定位：训练产物验收，不是数据集打标，也不是通用 SD WebUI。
- 默认路径：从“产物归档”的 checkpoint 或图像工作台工具入口进入。
- UI 组件：全部使用项目已有 shadcn 风格组件；缺失组件实现前用 shadcn CLI 补，不手写同类控件。
- 首屏主任务：用户打开页面后能直接选 LoRA、写 prompt、点生成。
- 参数完整但不拥挤：常用参数常驻，高级参数折叠。
- 结果优先：生成图固定尺寸网格展示，不让大图把页面撑高。
- 可复现：每张结果保存 prompt、negative、seed、steps、cfg、size、sampler、LoRA 权重、checkpoint 路径。

## 入口

### 图像工作台入口

在 `TOOLS` 增加一个工具：

- `id`: `lora-testbench`
- `category`: `ai`
- `stage`: `library` 或新增虚拟 stage `lora-test`
- `label`: `LoRA 生图测试`
- `requiresDataset`: `false`
- `async`: `true`
- `writes`: `true`

推荐新增虚拟 stage `lora-test`，避免把“训练后验收”塞进数据集生命周期的 `annotate/ship`。

### 产物归档入口

在 checkpoint 行旁增加“测试”按钮，跳转：

```text
/image-studio?stage=lora-test&job=<job_id>&checkpoint=<workspace-relative-path>
```

这样用户刚训练完可以从产物页直接测试，不需要手动复制路径。

## 页面布局

桌面布局采用三栏，只有中间结果区滚动：

```text
┌────────────────────────────────────────────────────────────┐
│ Header: LoRA 生图测试 · 当前 checkpoint · 状态 · 保存预设    │
├───────────────┬───────────────────────────┬────────────────┤
│ ModelPanel    │ PromptAndParamsPanel      │ QueuePanel     │
│ LoRA 选择      │ Prompt / Negative / 参数   │ 队列 / 日志      │
│ Base 模型      │ [生成] [加入队列] [随机种子] │ GPU / 失败原因   │
├───────────────┴───────────────────────────┴────────────────┤
│ ResultGrid: 固定卡片网格、对比、下载、复制参数、设为参考图     │
└────────────────────────────────────────────────────────────┘
```

移动端布局：

- 顶部：当前 LoRA + 生成按钮。
- Tabs：`模型`、`Prompt`、`参数`、`结果`。
- 队列用 `Sheet` 从底部打开。
- 结果图两列网格，详情用全屏 `Sheet`。

## shadcn 组件清单

优先复用现有组件：

- `Card`：模型选择、参数区、结果卡片。
- `Button`：生成、停止、下载、复制参数。
- `Input`：seed、尺寸、步数、cfg、LoRA 权重。
- `Textarea`：prompt、negative prompt。当前项目没有则实现前用 shadcn 添加。
- `Select`：checkpoint、sampler、scheduler、base model。
- `Tabs`：移动端和高级参数分组。
- `Switch`：随机 seed、保存 metadata、自动下载、启用负面词。
- `Slider`：LoRA weight、CFG、steps。当前项目没有则实现前用 shadcn 添加。
- `Sheet`：移动端队列、结果详情。
- `Dialog`：保存预设、覆盖确认。
- `Badge`：任务状态、后端类型、checkpoint 状态。
- `Progress`：单任务进度。
- `ScrollArea`：队列和结果详情局部滚动。
- `Skeleton`：加载 checkpoint/结果时占位。

不新增图表库，不新增表单库。

## 功能区

### 1. 模型选择

`ModelPanel` 负责选可生成的模型组合。

字段：

- `来源`: 最近产物 / 手动路径。
- `Job`: 从 `/api/artifacts` 的 jobs 列表选择。
- `Checkpoint`: 仅列出 `.safetensors/.sft` 的 LoRA 文件。
- `Base`: 从 job config snapshot 自动读取；允许手动覆盖。
- `Backend`: 自动识别 `anima_lora / kohya / diffusion-pipe`。
- `LoRA weight`: 默认 `1.0`，范围 `-2.0 ~ 2.0`。

规则：

- 从产物页进入时自动填 job 和 checkpoint。
- 如果 checkpoint 丢失，显示错误态，不显示空白表单。
- 如果 base model 路径缺失，允许用户手动选择，但生成按钮禁用直到路径有效。

### 2. Prompt 与参数

`PromptAndParamsPanel` 是主操作区。

常驻字段：

- `Prompt`
- `Negative prompt`
- `尺寸`: 预设 `768x1344 / 832x1216 / 912x1632 / 1024x1024`，支持自定义。
- `Seed`: `-1` 表示随机。
- `Batch count`
- `Batch size`
- `Steps`
- `CFG / guidance scale`
- `Sampler`

高级折叠字段：

- `Flow shift`
- `Scheduler`
- `Clip skip`
- `Precision`
- `Output format`: png/webp
- `Save metadata`
- `Output folder`
- `Prompt preset`

默认值：

- 尺寸默认从 job 的训练采样配置读取；没有则用 `912x1632`。
- steps 默认 `28`。
- CFG 默认 `4.5`。
- seed 默认 `-1`。
- batch count 默认 `4`。
- LoRA weight 默认 `1.0`。

### 3. 队列

生成必须走后台 session，不做一次性阻塞请求。

队列状态：

- `queued`
- `loading_model`
- `generating`
- `saving`
- `succeeded`
- `failed`
- `canceled`

队列展示：

- 当前 prompt 摘要。
- 进度：已生成 / 总数。
- 当前 seed。
- 用时。
- GPU 显存摘要。
- 错误 tail。
- `停止` 按钮。

刷新页面后，最近一次未完成任务必须恢复。

### 4. 结果网格

`ResultGrid` 是页面核心，不放进卡片套卡片。

每张结果卡展示：

- 图片，固定 aspect-ratio，`object-contain`。
- seed。
- checkpoint 名。
- steps / cfg / size / LoRA weight。
- 操作：放大、下载、复制参数、重新生成、作为参考。

详情 `Sheet`：

- 大图预览。
- 完整 prompt/negative。
- 完整参数 JSON。
- 文件路径。
- 下载按钮。

批量操作：

- 下载选中。
- 导出本次任务结果。
- 复制全部参数。
- 清空本次结果。

### 5. 预设

预设只保存 UI 参数，不保存实际模型权重。

保存内容：

- prompt
- negative
- size
- seed mode
- steps
- cfg
- sampler
- lora weight
- advanced params

不保存：

- checkpoint 绝对路径。
- base model 绝对路径。

预设入口：

- 保存当前参数。
- 从预设应用。
- 恢复默认。

## 后端 API 设计

复用现有推理能力，最小补接口。

### GET `/api/lora-test/models`

返回可选 LoRA：

```json
{
  "jobs": [
    {
      "job_id": "...",
      "output_name": "...",
      "backend": "anima_lora",
      "base_model": {...},
      "checkpoints": [
        {
          "path": "output/foo.safetensors",
          "size_bytes": 123,
          "modified_at": 1234567890
        }
      ]
    }
  ]
}
```

可直接复用 `/api/artifacts`，但新接口能带上 backend/base_model，前端少做二次请求。

### POST `/api/lora-test/generate`

启动生成 session：

```json
{
  "job_id": "...",
  "checkpoint_path": "output/foo.safetensors",
  "base_override": null,
  "prompt": "...",
  "negative_prompt": "...",
  "width": 912,
  "height": 1632,
  "seed": -1,
  "batch_count": 4,
  "batch_size": 1,
  "steps": 28,
  "cfg": 4.5,
  "sampler": "euler",
  "lora_weight": 1.0,
  "advanced": {}
}
```

返回：

```json
{ "session_id": "..." }
```

### GET `/api/lora-test/sessions/{id}`

返回任务状态、生成结果、错误。

### POST `/api/lora-test/sessions/{id}/cancel`

取消任务。已生成图片保留。

### GET `/api/lora-test/results/{id}/file?path=...`

返回单张图片，支持 inline 预览和下载。

## 推理实现边界

第一版只接入已经存在的 Anima LoRA subprocess 推理路径：

- 复用 `lorahub/core/inference/anima_lora_backend.py` 的 argv 构造思想。
- 输出放到 `runs/lora-test/<session_id>/`。
- 每张图写一个 sidecar JSON 保存参数。

不在第一版支持：

- ComfyUI 工作流编辑。
- img2img/controlnet。
- 多 LoRA 混合。
- 常驻模型热加载服务。

这些等基本测试页稳定后再加。

## 前端文件结构

```text
web/src/pages/image-studio/lora-test/
  index.tsx
  model-panel.tsx
  prompt-panel.tsx
  params-panel.tsx
  queue-panel.tsx
  result-grid.tsx
  result-detail-sheet.tsx
  presets.ts
  types.ts
```

注册点：

- `web/src/pages/image-studio/index.tsx`
- `web/src/pages/image-studio/tools-catalog.ts`
- `web/src/pages/image-studio/tool-registry.tsx`
- `web/src/pages/artifacts/index.tsx`

## 状态模型

URL 状态：

- `stage=lora-test`
- `job`
- `checkpoint`
- `session`

Local state：

- 表单草稿。
- 当前选中的结果。
- 详情 sheet 是否打开。
- 队列 sheet 是否打开。

localStorage：

- 最近一次参数。
- prompt 预设。
- 最近 session id。

## 验收标准

- 从产物归档点击任意 LoRA 的“测试”能进入页面并自动选中 checkpoint。
- 页面无 dataset 时仍可用。
- 生成任务刷新后不会丢状态。
- 结果网格不会被 912x1632 大图撑高。
- seed、prompt、cfg、steps、尺寸、LoRA weight 可在结果详情里完整追溯。
- 后端拒绝 workspace 外路径。
- 取消任务后已生成图片仍能查看。
- 移动端 390px 宽没有横向溢出。

## 实施顺序

1. 新增文档和页面路由占位。
2. 新增模型列表 API，先复用 artifact/job registry。
3. 新增 generate session API，只支持 Anima LoRA。
4. 实现 `lora-test` 页面三栏布局。
5. 接入队列轮询和取消。
6. 接入结果网格和详情 sheet。
7. 从产物归档加“测试”跳转。
8. 补移动端布局。
9. 跑后端路径安全测试和前端 build。

