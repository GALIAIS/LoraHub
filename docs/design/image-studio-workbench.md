# 图像工作台重写重构设计文档

Reading this as: 面向 LoRA 数据集制备的密集生产工作台，用户是反复处理成百上千张图片的训练用户，设计语言应是安静、稳定、低装饰的工具型产品 UI。

当前不是缺少一堆新概念，而是已经有不少后端能力，但前端把它们挤在工具卡片、弹窗、侧栏和阶段面板里。用户看到的是碎片，而不是“现在该做什么、点哪里开始、执行后怎么看结果”。重写目标是把核心能力做成首屏可见、主流程可操作的工作台，而不是继续往 `DatasetDetail` 里堆功能。

## 1. 设计边界

- 保留现有信息架构：`/image-studio?path=...&stage=...`，阶段仍是 `intake / audit / curate / annotate / ship`，`library` 继续作为跨数据集工具库。
- 保留现有视觉系统：继续使用项目已有 Tailwind、shadcn 风格组件和 lucide 图标，不新增设计系统依赖。
- 设计参数：`DESIGN_VARIANCE 3`，`MOTION_INTENSITY 2`，`VISUAL_DENSITY 8`。这是数据生产工具，不做营销页式大留白、渐变、装饰线、动画堆叠。
- 重写优先级：先拆状态和布局，再复用阶段工具，最后删掉重复入口和历史包袱。
- 页面优先级：核心功能直接露出，低频功能收进“更多”。不要让用户先理解工具广场、单工具页、弹窗和侧栏之间的关系。
- 本文是实现前设计文档，不在本次直接改 UI 代码。

## 2. 当前事实审计

### 2.1 前端结构

核心入口在：

- `web/src/pages/image-studio/index.tsx`
- `web/src/pages/image-studio/components/dataset-detail.tsx`
- `web/src/pages/image-studio/components/stages/*.tsx`
- `web/src/pages/image-studio/tools/*.tsx`
- `web/src/pages/image-studio/tools-catalog.ts`
- `web/src/lib/studio-task-store.ts`

现状页面有三套竞争入口：

- 左侧 `StudioSidebar` 选择数据集、阶段、工具库。
- `ToolsGrid` 和 `/image-studio/tools/:id` 提供工具广场和单工具页。
- 阶段页又把 `DatasetDetail` 和阶段 panel 拼在一起。

结果是用户进入 `audit / annotate / ship` 时，主屏仍先显示一个完整图片网格，真正当前阶段的功能被塞进底部或右侧面板。这个结构会天然拥挤，并让每个阶段都像附属插件，而不是独立工作流。

### 2.2 页面显示问题

这次重写优先解决显示和可用性问题：

- 首屏没有明确主任务。用户进来看到很多入口，但不知道先点哪个。
- 核心功能被藏起来。比如打标能力散在 `AnnotateStage`、`TaggingPanel`、`tools/tagging.tsx`、`tools/ai.tsx`、AI 批量弹窗里，用户无法在一个地方直接看到“WD14 打标 / 智能 caption / 触发词 / 词频 / 批改”。
- 页面密度高但信息层级弱。很多小卡片都像同级功能，真正高频动作没有更高优先级。
- 同一个能力有多个入口。工具广场、阶段面板、独立工具页、弹窗都能通向相似功能，用户会以为是不同东西。
- 阶段页不是阶段页。非 Curate 阶段仍被图库占据，当前阶段功能被压成附属区域。
- 结果不可见。任务开始后用户看到进度条，但不容易看到“写了多少 caption、哪些失败、下一步该检查什么”。
- 局部滚动不稳定。面板太多时底部内容不可达，或者滚动区域不符合用户预期。

### 2.3 `DatasetDetail` 的问题

`DatasetDetail` 现在同时负责：

- URL query 解析：`path / page / sort / recursive / view`
- 图片列表查询和详情查询
- 本地筛选、分页、排序
- 单选、多选、键盘快捷键
- 批量删除、收藏、导出路径复制
- 上传 drop zone
- AI 批量弹窗和任务条
- 标签面板、重复图视图、过滤面板
- Inspector、lightbox、pending ops dialog
- 删除确认和 toast

这个组件已经变成页面级应用容器，任何修改都容易影响其它工作流。它也导致 `curate` 以外的阶段不得不嵌套整个图库，造成重复滚动、内容挤压和移动端不可控。

### 2.4 列表和筛选问题

后端 `listings.py` 已经支持 `filter_caption / filter_quality / filter_aspect`，但前端 `applyFilters` 在当前页的 `data.items` 上做客户端筛选。后果：

- 当前页为空不代表全数据集无匹配。
- 总数和分页仍按未筛选列表计算。
- 子目录树只能基于当前页推导，不能代表完整数据集。
- 搜索、过滤、分页的语义不一致。

重写时应把能后端过滤的条件放到列表查询中，前端只保留轻量展示过滤或本页临时搜索。

### 2.5 滚动和大列表问题

`ImageGrid` 在图片数大于 500 时使用 `VirtuosoGrid`，但启用了 `useWindowScroll`。当前页面主体是内部 `overflow-y-auto` 容器，不是浏览器窗口滚动。这个组合容易造成虚拟列表高度、滚动位置、局部滚动不一致，也会影响移动端。

重写后应统一为工作区内局部滚动容器，虚拟列表绑定容器滚动，而不是 window scroll。

### 2.6 Inspector 问题

`Inspector` 固定 22rem 宽，负责预览、caption 编辑、AI 单图重标、触发词建议、构图分析、质量评分、人工评级、备注、旋转、翻转、删除等。它在桌面占据合理，但在阶段面板同时存在时会进一步压缩主内容。移动端应改为全屏 sheet，而不是侧边栏挤压。

### 2.7 任务状态问题

`studio-task-store.ts` 的方向是正确的：它把后台任务从页面挂载生命周期中拿出来，并用 localStorage 持久化 session id，解决刷新后任务条丢失的问题。

但当前只覆盖：

- `caption`
- `smart-caption`
- `wd14`
- `trigger-words`
- `quality-score`

而 `curate` 里的 `auto-rotate`、`batch-resize` 以及后续可能加入的相似度扫描等 session 没有统一进入这个 store。结果是任务条、取消、刷新恢复和错误显示仍会按功能分裂。

## 3. 后端能力审计

现有后端已经拆成多个子路由，不需要推倒重做：

- `datasets.py`：数据集 CRUD、上传、元数据。
- `listings.py`：图片列表、图片详情、基础服务端过滤。
- `intake.py`：源目录预检、本地路径导入、跨数据集复制，带 phash 去重。
- `audit.py`：审计扫描、报告缓存、问题列表。
- `dedupe.py`：感知哈希查重、聚类、批量删除。
- `similarity.py`：语义相似扫描、聚类、批量删除。
- `curate.py`：EXIF 自动旋转、隔离区、批量缩放、按 issue 处理、备份列表和恢复。
- `captions.py`：词频、查找替换、注入触发词、黑名单删除。
- `ai.py`：批量 caption、smart caption、质量评分、触发词抽取、单图改写。
- `tagging.py`：WD14/JoyTag 图像工作台打标任务，带 start/status/stop。
- `library.py`：全局标签、触发词、prompt 模板。
- `ops.py`：待执行操作队列和 apply。
- `ship.py`：训练前 lint、导出、另存数据集。

需要注意的真实缺口：

- `ship/export` 目前是 `StreamingResponse` 动态 zip，不具备稳定文件、断点续传和多格式能力。训练产物下载已有稳定缓存思路，但这里还没统一。
- `ops.py` 仍是小范围单图操作队列，不是所有 curate/annotate 写操作的统一事务层。
- 后端 task session 已经存在，但前端没有把所有 session 化任务纳入一个 lifecycle。
- 列表 API 的搜索、子目录、完整服务端过滤还不够完整。

## 4. 产品模型

图像工作台应按“数据集生命周期”组织，而不是按组件和工具散落：

1. `Intake`：把图安全导入数据集。
2. `Audit`：扫描并解释当前数据集质量。
3. `Curate`：围绕图片网格做筛选、挑选、删除、修正、缩放、隔离。
4. `Annotate`：维护 caption、标签、触发词、AI 标注任务。
5. `Ship`：训练前检查、导出、另存、跳转训练配置。

每个阶段都应有自己的主视图。图片网格不是所有阶段的默认主体，只在 `curate` 是核心，在其它阶段作为可打开的“查看样本”或“定位问题”辅助视图。

每个阶段只能有一个主动作区：

- `Intake` 主动作是导入。
- `Audit` 主动作是扫描和查看问题。
- `Curate` 主动作是看图、选择、处理。
- `Annotate` 主动作是打标和批量改 caption。
- `Ship` 主动作是检查和导出。

工具广场只做快捷入口，不再承担主流程。用户不应该为了打标先去猜“打标属于工具、AI、标注还是侧栏”。

## 5. 目标布局

### 5.1 桌面布局

工作台分三层，不再把阶段 panel 嵌进图库：

```text
┌─────────────────────────────────────────────────────────────┐
│ DatasetRail  数据集、创建、全局库、任务入口                  │
├─────────────────────────────────────────────────────────────┤
│ DatasetHeader  当前数据集、数量、caption 覆盖、lint 状态      │
├─────────────────────────────────────────────────────────────┤
│ StageNav  Intake | Audit | Curate | Annotate | Ship          │
├─────────────────────────────────────────────────────────────┤
│ StageWorkspace                                               │
│   当前阶段的主任务区                                         │
└─────────────────────────────────────────────────────────────┘

Inspector / TaskConsole / BulkPreview 作为右侧 drawer 或底部 drawer 打开。
```

关键规则：

- `StageWorkspace` 只有一个主滚动容器。
- `DatasetHeader` 和 `StageNav` 固定在工作区顶部，但高度必须紧凑。
- `Inspector` 不常驻挤压所有页面。桌面为右侧 drawer，移动端为全屏 sheet。
- `TaskConsole` 是全局底部抽屉，显示所有当前数据集相关任务。
- 工具广场不再和阶段视图竞争。工具卡片只作为 deep link，落点回到对应阶段的某个 panel。

### 5.2 移动端布局

移动端不能压缩桌面三栏。目标结构：

- 顶部一行：数据集选择 + 当前任务状态。
- 第二行：阶段 segmented control，可横向滑动。
- 主体：当前阶段单列内容。
- 底部 sticky action bar：选择数、批量操作、打开筛选、打开任务。
- Inspector、筛选、批量预览都用全屏 sheet。

移动端不显示常驻侧栏，不显示左右双 panel，不让图库和阶段工具同屏互相挤压。

## 6. 阶段设计

### 6.1 Intake

目标：导入前先知道会发生什么。

主视图：

- 上传 drop zone。
- 本地路径导入。
- 跨数据集复制。
- 预检结果：总数、可导入、已存在、批内重复、sidecar 状态。
- 导入结果：新增、跳过、失败、失败原因。

复用接口：

- `datasetUpload`
- `imageStudioIntakePreflight`
- `imageStudioIntakeLocalPath`
- `imageStudioIntakeFromDataset`

设计要求：

- 预检和执行分开，执行前显示影响范围。
- 预检结果可以直接进入 Curate 的筛选视图。
- 失败列表可复制，不用 toast 承载长错误。

### 6.2 Audit

目标：不修改数据，只给出可行动的问题地图。

主视图：

- 顶部 summary：图片数、caption 覆盖、触发词覆盖、阻塞问题、警告问题。
- 问题分组：损坏、极端比例、缺 caption、重复 caption、重复图、近似重复。
- 分布图：尺寸、比例、文件大小、caption 长度。
- 问题行的操作不是直接删除，而是“查看样本”“加入 Curate 队列”“重新扫描”。

复用接口：

- `imageStudioAuditScan`
- `imageStudioAuditReport`
- `imageStudioDedupeScan`
- `imageStudioDedupeClusters`
- `imageStudioSimilarityScan`
- `imageStudioSimilarityClusters`

设计要求：

- Audit 页面不常驻大图库。
- 点击问题后打开局部样本 drawer 或跳转 Curate 并带过滤条件。
- 扫描状态进入统一任务条。

### 6.3 Curate

目标：图片选择和文件修正的主工作区。

主视图：

- 左侧或顶部筛选条：caption、质量、比例、子目录、收藏、问题来源。
- 中央虚拟图片网格。
- 底部选择操作栏：删除、收藏、隔离、AI、导出、清空选择。
- 右侧 Inspector drawer：只在选中图片时打开。
- Dedupe/Similarity 结果以“聚类审查模式”进入，不另起一个脱离网格的页面。

复用接口：

- `imageStudioList`
- `imageStudioGetImage`
- `imageStudioSaveAnnotation`
- `imageStudioAddOp`
- `imageStudioApplyOps`
- `imageStudioBatchDelete`
- `imageStudioAutoRotate`
- `startImageStudioAutoRotate`
- `imageStudioBatchResize`
- `startImageStudioBatchResize`
- `imageStudioQuarantine`
- `imageStudioRestoreQuarantine`
- `imageStudioBackupsList`
- `imageStudioRestoreBackup`

设计要求：

- 服务端过滤优先，分页总数必须可信。
- 虚拟列表绑定局部滚动容器。
- 删除默认走隔离或 trash，永久删除必须二次确认。
- 批量操作都进入“预览影响 -> 执行 -> 结果”流程。

### 6.4 Annotate

目标：打标是图像工作台的主要功能。它必须是一个成熟的“Caption Studio”，用户进入页面后可以直接看覆盖率、选打标方式、启动任务、看结果、复核失败、批量修正，而不是在工具页、AI 弹窗、侧栏之间找功能。

主视图：

- 顶部 `Caption Health`：总图片、已有 caption、缺失 caption、过短 caption、触发词覆盖、最近一次打标状态。
- 左侧 `Mode`：打标方式选择，不用工具卡片网格。
  - `自动标签`：WD14 / JoyTag，适合 booru tag 数据集。
  - `智能 Caption`：WD14 + VLM/LLM 两步，适合 Anima 风 caption。
  - `视觉 Caption`：VLM 直接看图写 caption，适合自然语言描述。
  - `质量/触发词`：质量评分和触发词抽取，作为辅助模式。
- 中间 `Run Setup`：当前模式参数，默认只露出 5 个以内高频项。
- 右侧 `Preview`：执行前抽样展示 6-12 张图片和将要写入的 caption 形态。
- 底部 `Run Result`：本次任务的写入、跳过、失败、耗时、失败列表和复核入口。
- 下方 `Caption Maintenance`：词频、查找替换、注入触发词、黑名单删除、低频 tag 清理。

首屏布局：

```text
┌─ Caption Health ────────────────────────────────────────────┐
│  812 images  624 captioned  188 missing  trigger 71%  last ok │
└──────────────────────────────────────────────────────────────┘
┌─ Mode ─────────────┬─ Run Setup ─────────────────┬─ Preview ─┐
│ 自动标签            │ 模型 / 范围 / 合并 / 触发词   │ 抽样图片   │
│ 智能 Caption        │ [开始打标] [预览影响]         │ caption样例│
│ 视觉 Caption        │ 高级设置折叠                  │            │
│ 质量/触发词         │                              │            │
└────────────────────┴──────────────────────────────┴──────────┘
┌─ Run Result / Review Queue ─────────────────────────────────┐
│  写入 612  跳过 188  失败 12  [查看失败] [抽样复核] [进入Ship] │
└──────────────────────────────────────────────────────────────┘
```

模式设计：

- `自动标签`
  - 默认模型：项目已有 WD14/JoyTag 配置。
  - 高频参数：模型、阈值、覆盖范围、跳过已有 caption、合并策略。
  - 输出：tag 列表，支持 append/replace/prepend。
  - 结果复核：显示高频 tag、低频 tag、空结果、失败图片。
- `智能 Caption`
  - 默认流程：WD14 取 tags，再由 VLM/LLM 生成 Anima 风 caption。
  - 高频参数：训练用途、触发词、图片是否上传给 VLM、合并策略、跳过已有。
  - 输出：适合训练的 caption 文本。
  - 结果复核：展示原 caption、新 caption、变更摘要。
- `视觉 Caption`
  - 默认流程：VLM 直接看图写 caption。
  - 高频参数：prompt 模板、语言风格、最大长度、覆盖范围、合并策略。
  - 输出：自然语言或半结构化 caption。
  - 结果复核：重点看过长、空响应、重复响应。
- `质量/触发词`
  - 辅助打标，不抢主流程。
  - 输出写入 annotation，不默认覆盖 caption。
  - 结果用于 Curate 过滤和 Ship lint。

范围选择：

- 默认范围是 `缺失 caption`，这是最符合新用户预期的安全默认值。
- 可选范围：全部图片、当前筛选、当前选中、缺失 caption、低质量 caption、失败重试。
- 选择“全部图片”且会覆盖现有 caption 时，必须显示影响数量和确认。

结果复核：

- 任务完成后页面不只弹 toast，必须停留在 `Run Result`。
- 结果分组：`已写入`、`已跳过`、`失败`、`需要复核`。
- `需要复核` 的来源：空 caption、过短、过长、重复 caption、模型返回错误、触发词缺失。
- 抽样复核一次显示图片、原 caption、新 caption、操作按钮：保留、回滚、编辑、加入黑名单。

批量维护：

- 词频不是单独工具页，而是维护区的第一项。
- 用户可以从词频表选择 tag 后直接执行：删除、替换、加入黑名单、加入触发词规则。
- 查找替换必须有 preview，不直接写。
- 注入触发词要显示命中数量、已存在数量、将修改数量。
- 黑名单删除完成后自动刷新词频和 caption health。

失败恢复：

- 每个打标任务有 session id、参数快照和结果摘要。
- 失败后提供 `重试失败项`，不要求用户重新跑整个数据集。
- 页面刷新后仍显示未完成或最近完成的打标任务。
- 取消任务后保留已写入结果，并显示“已取消，写入 N 张，剩余 M 张”。

移动端：

- `Caption Health` 横向滚动指标条。
- `Mode` 变成顶部 segmented control。
- `Run Setup` 单列展示。
- `Preview` 和 `Run Result` 用折叠区，结果完成后自动展开。
- 批量维护区默认折叠，避免压住主打标操作。

复用接口：

- `startTaggingSession`
- `imageStudioCaptionsVocab`
- `imageStudioCaptionsFindReplace`
- `imageStudioCaptionsInjectTrigger`
- `imageStudioCaptionsBlacklist`
- `startCaptionSession`
- `startSmartCaptionSession`
- `startQualitySession`
- `startTriggerWordsSession`
- `imageStudioSmartCaptionSingle`

设计要求：

- 用户进入 `Annotate` 首屏就能看到打标按钮，不需要打开工具广场或 AI 弹窗。
- WD14、智能 caption、VLM caption 是同一个“打标中心”的三个模式，不再散落成多个看似无关的页面。
- 低频能力如单图 WD14 测试、VLM Anima 重写放到“调试工具”，不抢主流程。
- 默认展示原始 caption 和变更预览，不用多个小卡片堆满页面。
- 批量写入必须有 dry-run 或影响预览。
- AI 任务的进度、取消、刷新恢复统一进 TaskConsole。
- 任务完成后停留在结果区，明确给出下一步：查看失败、抽样复核、进入 Ship 检查。

### 6.5 Ship

目标：训练前最后确认和交付。

主视图：

- Lint gate：ready、blockers、warnings、stale reason。
- 导出：全量、选中、排除隔离区、包含元信息。
- 另存数据集。
- 后续入口：创建训练任务并预填 dataset path。

复用接口：

- `imageStudioShipLint`
- `imageStudioShipExport`
- `imageStudioShipSaveAs`

需要补齐：

- `ship/export` 改为稳定归档文件，支持 zip、tar、tar.gz、tar.xz，返回可断点续传下载链接。
- 导出大数据集时应有后台任务和历史记录，不应只靠一次 fetch 流。

## 7. 状态和数据模型

### 7.1 URL 状态

建议集中成 `useImageStudioUrlState`：

- `datasetPath`
- `stage`
- `tool`
- `page`
- `sort`
- `recursive`
- `filters`
- `selectedPath`
- `selectionMode`

规则：

- 可分享和可恢复的状态放 URL。
- 临时 UI 状态放组件本地，例如 drawer 是否打开。
- 选择集较大时不进 URL，只保存在 stage store。

### 7.2 图片列表查询

建议集中成 `useDatasetImages`：

- 输入：dataset path、page、limit、sort、recursive、server filters、search、subdir。
- 输出：items、total、page、loading、error、empty reason。
- 统一 query key，避免全项目散落 `["image-studio"]` 粗粒度 invalidate。

后端需要补齐：

- `search`
- `subdir`
- `favorite`
- `issue_kind`
- `has_trigger`

### 7.3 选择状态

建议集中成 `useImageSelection`：

- 单选和多选分离。
- 支持 select page、select filtered query、clear、invert page。
- 批量操作默认只作用于明确选择集。
- “选中当前筛选全部”必须通过后端确认数量，不能只选当前页。

### 7.4 Inspector 状态

建议拆成：

- `InspectorDrawer`
- `ImagePreview`
- `CaptionEditor`
- `ImageMetaPanel`
- `ImageOpsPanel`

Inspector 只关注当前图片，不持有列表和批量任务状态。

### 7.5 任务生命周期

保留 `studio-task-store.ts` 的思路，扩成统一 `ImageStudioTaskStore`：

任务类型至少覆盖：

- `caption`
- `smart-caption`
- `wd14`
- `trigger-words`
- `quality-score`
- `auto-rotate`
- `batch-resize`
- `similarity-scan`
- 后续新增的导出任务

每类任务定义：

- `start`
- `status`
- `cancel`
- `mapStatus`
- `label`
- `resultInvalidation`

UI 只消费统一记录：

- `queued / running / succeeded / failed / canceled`
- `processed / total / percent`
- `lastImage`
- `error`
- `events`
- `canCancel`

### 7.6 打标中心状态

`Annotate` 不应把每个打标能力拆成独立页面状态。建议一个轻量本地状态：

- `mode`：`auto-tags / smart-caption / vlm-caption / quality-trigger`
- `scope`：`missing / all / filtered / selected / failed`
- `mergeStrategy`：`replace / append / prepend`
- `triggerWord`
- `skipExisting`
- `advancedOpen`
- `lastRunId`
- `reviewGroup`：`written / skipped / failed / needs-review`

规则：

- `mode / scope / lastRunId` 可进 URL，便于刷新和分享。
- 具体表单草稿留在组件本地。
- 任务结果由 TaskConsole 和后端 session/status 提供，不在多个组件重复存。

## 8. 组件重构方案

目标文件结构：

```text
web/src/pages/image-studio/
  index.tsx
  state/
    use-image-studio-url-state.ts
    use-dataset-images.ts
    use-image-selection.ts
    use-stage-task-console.ts
  layout/
    image-studio-shell.tsx
    dataset-header.tsx
    stage-nav.tsx
    task-console.tsx
  stages/
    intake/
    audit/
    curate/
    annotate/
    ship/
  shared/
    image-grid-pane.tsx
    image-tile.tsx
    inspector-drawer.tsx
    bulk-action-bar.tsx
    impact-preview-dialog.tsx
    empty-state.tsx
    error-state.tsx
```

拆分规则：

- `index.tsx` 只做路由状态和 shell 组合。
- `DatasetDetail` 退役，不再作为所有阶段的根组件。
- `ImageGrid` 改为 `ImageGridPane`，只管展示和选择，不管 URL、查询、删除、AI。
- `Inspector` 改为 drawer，独立管理单图编辑。
- `tools/*.tsx` 里的可用 panel 迁移进对应 `stages/*`，工具页只作为跳转和高亮入口。
- `DatasetManager` 如果已不在主路径使用，应在重构后删除或并入 `DatasetRail`。

## 9. 交互规范

- 高频功能放主区，低频功能折叠。不要用工具卡片网格平铺所有能力。
- 每个阶段首屏必须有一个明确主按钮或主列表。用户不应先阅读说明才能知道怎么开始。
- 同类功能合并成模式切换。比如打标是一个中心，WD14/智能 caption/VLM caption 是模式，不是三个入口。
- 所有写操作走同一形态：选择范围、预览影响、执行、结果、可恢复入口。
- 所有长任务走 TaskConsole，不再在各组件里各自放轮询和状态条。
- 所有空态说明下一步可执行动作。
- 所有错误态显示具体失败项，不把长错误塞进 toast。
- toast 只用于短反馈，不承担报告。
- 常驻页面不显示无关工具卡片，避免“功能很多但不知道做什么”的臃肿感。
- 图像预览必须有固定尺寸和 `object-contain`，不能把布局撑高。
- 表单和工具面板使用局部滚动，页面根不出现双重不可控滚动。

## 10. 后端小改清单

不建议大改后端，优先补下面几处：

1. 扩展 `/image-studio/list`：支持 search、subdir、favorite、issue kind，并返回过滤后的 total。
2. 统一 task session 描述：给 auto-rotate、batch-resize、similarity、export 暴露一致 start/status/stop 响应字段。
3. 改造 `/image-studio/ship/export`：后台生成稳定归档文件，复用 FileResponse，支持断点续传和 zip/tar/tar.gz/tar.xz。
4. 扩展 ops 或新建 batch mutation 层：对批量写操作提供 dry-run、apply、result。
5. 审计报告返回可跳转过滤条件，例如 `issue_kind`、`paths_sample`、`count`。

## 11. 实施顺序

### F0 文档和保护线

- 保留本设计文档。
- 列出要退役的组件和要复用的 API。
- 先不删旧入口，避免中途不可用。

### F1 Shell 和 URL 状态

- 新建 `ImageStudioShell`、`DatasetHeader`、`StageNav`。
- 新建 `useImageStudioUrlState`。
- 页面仍可进入旧 `DatasetDetail`，但 stage 容器先换成新 shell。

### F2 Curate 主工作区

- 新建 `useDatasetImages` 和 `useImageSelection`。
- 新建 `ImageGridPane`，修复虚拟列表局部滚动。
- 新建 `InspectorDrawer`。
- 将删除、收藏、旋转、翻转、lightbox、批量选择迁入新结构。

### F3 Intake/Audit/Annotate/Ship 阶段化

- 把现有 stage panel 提升为各阶段主视图。
- 工具页 deep link 回阶段视图并高亮对应 panel。
- 移除“非 Curate 阶段仍常驻图库”的布局。
- 优先做 `Annotate` 的打标中心，把 WD14、智能 caption、VLM caption 聚合到首屏。
- 打标中心先只复用已有接口，不新增后端算法。成熟度来自页面组织、默认范围、结果复核和失败重试。

### F4 TaskConsole

- 扩展 `studio-task-store.ts` 的任务类型。
- 把 WD14、AI caption、quality、trigger、auto-rotate、batch-resize 统一显示。
- 刷新后恢复、取消、失败、完成结果都走同一 UI。

### F5 服务端过滤和批量影响预览

- 列表过滤下沉后端。
- 批量 find/replace、blacklist、quarantine、resize 加影响预览。
- 选择当前筛选全部时由后端返回数量和确认摘要。

### F6 移动端专项

- 单列 stage workspace。
- Inspector、Filter、TaskConsole 全屏 sheet。
- 底部 sticky action bar。
- 用真实示例任务验证 360px、390px、430px 宽度。

### F7 清理旧代码

- 删除或合并 `DatasetDetail` 的旧职责。
- 删除重复工具页占位和重复入口。
- 清理粗粒度 query invalidation。

## 12. 验收标准

桌面：

- 1000 张图片的数据集滚动稳定，无 window scroll 虚拟列表错位。
- `audit / annotate / ship` 不再显示“图库 + 挤压面板”的双主视图。
- 筛选后的 total、分页、空态准确。
- Inspector 打开不会让主阶段布局崩塌。
- 批量任务刷新页面后仍能看到状态。

移动端：

- 没有横向溢出。
- 主要操作可在单列完成。
- Inspector 和任务台以 sheet 方式打开。
- 底部操作栏不遮挡关键内容。

数据安全：

- 写操作有影响预览或确认。
- 删除可从 trash/quarantine/backup 找回。
- 大导出可恢复下载，不依赖一次性 fetch。

代码维护：

- 单个页面组件不再超过约 300 行。
- 查询、选择、任务、Inspector 各有明确边界。
- 新增一个工具只需要注册到工具目录和对应 stage，不需要改多个入口。
