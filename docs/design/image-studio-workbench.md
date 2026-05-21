# 图像工作台重构 — 真正能干活的数据集处理流水线

> 否定上一份方案。当前工作台是"相册 + 打标"皮,做不了真正的数据集准备工作。
> 这份重写按训练数据集制备的实际工作流来切分,补全缺失能力。

---

## 1. 从训练实际需要什么倒推 — 缺什么

训练一个高质量 LoRA,数据集要经历的工序(按经验顺序)：

| # | 工序 | 现状 | 缺什么 |
|---|---|---|---|
| 1 | **导入** — 从 zip / 多源目录拖图,自动去重 | ✅ 上传 zip | 无差异感知导入(同图重复添加无提示) |
| 2 | **审计** — 看清楚有什么:分辨率分布 / 文件大小 / EXIF / 损坏文件 | ❌ 仅缩略图 | **整个审计层缺失** |
| 3 | **质量过滤** — 美学打分 / 模糊检测 / 噪点 / NSFW / 黑白 / 纯色 | ❌ 接了 quality 端点但只输出整体分数 | **没有维度化的质量信号** |
| 4 | **去重** — 像素级 + 语义级(找几乎重复但不全等) | ✅ phash + similarity 后端 | similarity 没接前端 |
| 5 | **裁剪 / 修正** — 智能裁主体 / 去黑边 / 居中 / face-crop / 旋转 EXIF | ❌ 仅 90° rotate / flip | **裁剪管线全缺** |
| 6 | **分辨率规范化** — 按训练 bucket 重采样,丢小图 | ❌ 训练时才做(在 anima 内部) | 工作台看不到也调不了 |
| 7 | **遮罩 / 掩码** — masked loss 用的 alpha mask | ❌ 后端 anima_lora 有 generate_masks 脚本 | **完全没接前端** |
| 8 | **打标** — 自动 tag / 智能 caption | ✅ WD14 + AI VLM | 缺 tag 词表统计 / 全局批改 |
| 9 | **标签管理** — 频率分布 / 全局替换 / 触发词强化 / 黑名单 | ❌ 完全没有 | **整个标签后处理缺失** |
| 10 | **平衡 / 采样权重** — 调子集比例(角色 A:B = 3:1) | ❌ schema 有 subsets / num_repeats 但工作台不调 | **数据平衡缺失** |
| 11 | **训练前检查** — 缺 caption / 触发词不一致 / 极端 AR | ❌ 训练时才报 | **没有前置 lint** |
| 12 | **导出** — 真正打 zip / 同步到 VPS / 复制为新数据集 | ❌ "导出"只是 copy paths | **真导出缺失** |
| 13 | **版本快照** — 标签批改前后能看 diff,能回滚 | 部分 (ops 队列只覆盖 6 种 op) | 缺 dataset 级 snapshot |

**结论**：当前能做的只有 #1 #4(部分)#8,共 3/13。其他 10 项要么完全没有,要么后端有壳但前端没暴露,要么停留在 toy demo 程度。

---

## 2. 重新分块:5 个工作流域

抛弃"按视图分 tab"思路。按**用户来工作台是想完成什么** 切 5 个域。每域是一个独立的功能集合,可以包含多个子功能,但首页要有清晰入口。

```
┌─ Dataset Workbench ──────────────────────────────────────────┐
│                                                              │
│   📥 Intake     →  📊 Audit    →  ✂️  Curate   →  🏷  Annotate  →  🚀 Ship
│   导入 / 摄取      审计 / 体检      整理 / 编辑     标注 / 标签     输出 / 训练
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

每个域是一个 stage,不是孤立 tab。完成一个 stage 的状态会影响下一个 stage 是否可用(如 Annotate 没完成时 Ship 显示 ⚠ blocked)。

### 2.1 Intake — 导入 / 摄取

**目标**:把图弄进数据集。比当前多 5 件事。

| 能力 | 现状 | 新做 |
|---|---|---|
| 拖入图片 / zip | ✅ | 保留 |
| 拖入目录(浏览器原生 directory upload) | ❌ | 新增 — 整个文件夹层级一次拽进 |
| 从外部路径吸附 | ❌ | 新增 — 输入服务器上的本地路径,后端 `os.scandir` 收 |
| 从已有数据集复制(子集 / 全部) | ❌ | 新增 — 选源数据集 + glob,COW 复制 |
| URL / civitai 数据集导入 | ❌ | 后期(需自动下载模块) |
| 导入时去重 | ❌ | 新增 — 摄取前算 phash,与已有库比对,提示"5 张已存在" |
| EXIF 去敏 + 自动旋转 | ❌ | 新增 — 自动应用 EXIF orientation,写入新 JPEG |

**后端新增**:
- `POST /api/image-studio/intake/local-path` — 从服务器路径拷入
- `POST /api/image-studio/intake/from-dataset` — 跨数据集复制
- `POST /api/image-studio/intake/preflight` — phash dedup 预检

### 2.2 Audit — 审计 / 体检

**目标**:看清楚数据集有什么问题,不下手改。是 dashboard 加诊断。

| 维度 | 现状 | 新做 |
|---|---|---|
| 总数 / caption 覆盖率 / 收藏数 | ❌ | 新增 — Overview 卡片 |
| 分辨率直方图(短边 / 长边 / AR) | ❌ | 新增 — recharts 直方图 |
| 文件大小分布 + 可疑大文件 | ❌ | 新增 |
| EXIF orientation 异常 | ❌ | 新增 — 没自动旋转的红标 |
| 损坏图(无法打开 / 0 字节 / 截断 PNG) | ❌ | 新增 — Pillow 校验 |
| 颜色:黑白 / 纯色 / 偏色 | ❌ | 新增 — 直方图熵 + 标准差 |
| 美学评分 (Schuhmann aesthetic predictor) | ❌ | 新增 — open_clip + linear head |
| NSFW 评分 | ❌ | 新增 — wd-tagger 已有 nsfw 通道,聚合输出 |
| 模糊检测 (Laplacian variance) | ❌ | 新增 — OpenCV 一行 |
| Caption 长度分布 + 异常短 caption | ❌ | 新增 |
| 标签频率词云 + 长尾(出现 1 次的 tag) | ❌ | 新增 — 与 Annotate 共享同份索引 |
| 触发词覆盖一致性 | ❌ | 新增 — 选定触发词,扫多少张图缺它 |
| 重复 / 近似重复簇数 | ✅(后端) | 接前端 |

**后端新增**:
- `POST /api/image-studio/audit/scan` — 一次性算上面所有,缓存到 `audit_cache.json` (per dataset)
- `GET /api/image-studio/audit/report` — 拿缓存 + 增量更新
- `POST /api/image-studio/audit/aesthetic` — 单独触发美学打分(慢,放后台)

### 2.3 Curate — 整理 / 编辑

**目标**:按审计结论批量动手改图。

| 能力 | 现状 | 新做 |
|---|---|---|
| 单图 90° 旋转 / 翻转 | ✅ | 保留 |
| 单图删除 / 收藏 | ✅ | 保留 |
| 智能裁剪到训练 AR(face / saliency-aware) | ❌ | 新增 — yolo-face / U2Net 主体居中裁 |
| 居中裁 / 顶部裁 / 底部裁(批量) | ❌ | 新增 — 简单几何裁,带预览 |
| 去黑边(letterbox 自动检测) | ❌ | 新增 |
| 自动旋转 EXIF | ❌ | 新增 — Audit 报警 → Curate 一键修 |
| 按 Audit 维度批量删除(模糊 / 损坏 / 黑白 / NSFW) | ❌ | 新增 — "Audit 选中模糊 < 50 → 一键删 / 移到隔离区" |
| 上采 / 下采到目标分辨率(BasicSR / Lanczos) | ❌ | 新增 — 训练前规范化 |
| 移到隔离子目录(_quarantine/) | ❌ | 新增 — 不直接删,用户犹豫的图先放隔离区 |
| 真正的语义聚类 + 簇内批删 | ✅(后端 similarity) | 接前端 |

**后端新增**:
- `POST /api/image-studio/curate/auto-crop` — 智能裁剪批量任务
- `POST /api/image-studio/curate/auto-rotate` — EXIF 修正
- `POST /api/image-studio/curate/quarantine` — 移到隔离区
- `POST /api/image-studio/curate/restore` — 从隔离区拿回
- `POST /api/image-studio/curate/resize` — 批量重采样

**关键设计**:Curate 的所有改动**默认走 ops 队列**,不立即落盘。隔离区是"逻辑删除"的目录形式,Apply ops 时才真删。

### 2.4 Annotate — 标注 / 标签

**目标**:打 caption 不只是"跑个 WD14",还要管理标签。

| 能力 | 现状 | 新做 |
|---|---|---|
| WD14 自动打标 | ✅ | 保留,但 diff 化 |
| AI VLM caption(Claude / GPT-4o / Gemini) | ✅ | 保留,放整页 |
| Anima Tagger(就在 anima_lora 里!) | ❌ | 新增 — 走 `library.captioning.anima_tagger` |
| **标签词表 + 频率统计** | ❌ | 新增 — 整个数据集 tag 直方图 |
| **全局批改** — 找 "1girl" → 替 "1 girl" / 删除 / 添加 | ❌ | 新增 — 类似 IDE 的 find-replace |
| **触发词强制注入** — "所有图开头加 @char_a" | ❌ | 新增 |
| **触发词检查** — 显示哪些图缺触发词 | ❌ | 新增 |
| **caption 长度调节** — 截断到 N tags / 补充到至少 M tags | ❌ | 新增 |
| **黑名单标签** — 删除一组 tag(如 nsfw 类) | ❌ | 新增 |
| **大小写 / 下划线规范化** | ❌ | 新增 — 一键 lowercase + 替空格 |
| **手工编辑 + 提示** — caption 编辑器,自动补全已有 tag | ✅(基础)| 加补全 |
| **撤销 / 历史** | 部分 | 新增 — caption 级别 diff history |

**后端新增**:
- `POST /api/image-studio/captions/find-replace` — 全局查改,返回 dry-run 结果
- `POST /api/image-studio/captions/normalize` — 一键规范化(小写 / 去空格 / 去重 tag)
- `POST /api/image-studio/captions/inject-trigger` — 触发词批注入
- `POST /api/image-studio/captions/blacklist` — 黑名单 tag 删除
- `GET /api/image-studio/captions/vocab` — tag 频率词表
- `POST /api/image-studio/captions/anima-tagger` — 接 anima_tagger 路径(已有 model)

### 2.5 Ship — 输出 / 训练

**目标**:数据集就绪,送去训练或导出。

| 能力 | 现状 | 新做 |
|---|---|---|
| **训练前 lint 报告** — 缺 caption / 短边 < bucket / 触发词缺失 / NSFW > 阈值 | ❌ | 新增 — 红黄绿灯 + 详情 |
| **直接新建训练任务** — 跳到 Configs 页,数据集预填,推荐 8gb / 32gb 配置 | ❌ | 新增 |
| **导出为 zip / tar.gz** — 真正打包下载 | ❌ | 新增 |
| **同步到 VPS** — 通过现有 SSH 配置,rsync / scp | ❌ | 新增(高价值) |
| **保存为新数据集** — 当前过滤 / 子集另存 | ❌ | 新增 |
| **一键复制结构(空数据集)** — 新数据集复用同一份训练触发词 / blacklist 配置 | ❌ | 新增 |

**后端新增**:
- `GET /api/image-studio/ship/lint` — 训练前检查,返回 issues 列表
- `POST /api/image-studio/ship/export` — 打 zip,流式下载
- `POST /api/image-studio/ship/sync-remote` — rsync 到远端
- `POST /api/image-studio/ship/save-as` — 当前筛选另存为新数据集

---

## 3. 数据集级 snapshot 系统(横切)

工作台所有动手的步骤(Curate / Annotate / 删除)默认创建一个 snapshot,即"操作前的 caption + 文件清单"。

```
datasets/<name>/.workbench/snapshots/
  20260522-1530-pre-find-replace.json
  20260522-1545-pre-quarantine-blurry.json
```

每个 snapshot 是一个 `{path: caption, files: [...]}` 索引。占空间小(几百 KB),恢复快。

新端点:
- `POST /api/image-studio/snapshots/create`
- `GET /api/image-studio/snapshots/list`
- `POST /api/image-studio/snapshots/restore/{id}`

UI 入口:每个 stage 顶部有"创建快照"按钮,Snapshot 浏览器在工作台底部 history-bar(默认折叠)。

---

## 4. 现有 ops 队列升级

ops 当前只 6 种。扩到所有 Curate / Annotate 动作都走队列：

| 现 op | 新增 op |
|---|---|
| rotate | crop, smart_crop, resize, exif_rotate |
| flip | quarantine, restore_quarantine |
| replace_caption | find_replace_caption, normalize_caption, inject_trigger, blacklist_tags |
| merge_caption |  |
| favorite |  |
| delete |  |

每个 op 必须实现 `apply` + **`reverse`**(撤销),让 ops 队列真正成为安全网。

---

## 5. UI 主结构

5 个 stage 横向 stepper 在顶部,内容区按当前 stage 切换。每 stage 是一个"小应用",可以有自己的子 tab：

```
┌─ Top Bar ──────────────────────────────────────────┐
│ ← 数据集列表  │ azurlane_char0 ▾  │ ⚠ 16 缺 caption │
└────────────────────────────────────────────────────┘
┌─ Stage Stepper ────────────────────────────────────┐
│ ① Intake   ②⊘ Audit   ③ Curate   ④● Annotate   ⑤ Ship │
│   完成        待开始       进行中       49/111 完成     就绪 │
└────────────────────────────────────────────────────┘
┌─ Stage Body(变内容)─────────────────────────────┐
│                                                    │
│ (各 stage 自己的视图)                              │
│                                                    │
└────────────────────────────────────────────────────┘
┌─ History / Snapshots(底部抽屉,默认折叠)─────────┐
│ ▸ 12:34  Curate: 删除 7 张模糊图     [回滚]       │
│ ▸ 12:30  Annotate: WD14 标注全部     [回滚]       │
└────────────────────────────────────────────────────┘
```

**stage 状态指示灯**(绿/黄/红)由 audit 报告 + 当前数据集 meta 计算,引导用户走完整个流程。

---

## 6. 核心补全清单(优先级 P0)

按"做完一遍数据集处理需要什么"列最低必需补的,这是从 toy 到能用的最小集：

1. **Audit 全维度报告**(分辨率/文件大小/损坏/模糊/美学/NSFW/caption 覆盖率)
2. **智能裁剪到训练 AR**(face-aware + saliency-aware 二选一)
3. **批量重采样到目标分辨率**
4. **隔离区**(逻辑删除,可恢复)
5. **标签词表 + find-replace + 触发词批注入 + 黑名单**
6. **训练前 lint 报告**
7. **真导出 zip + 同步 VPS**
8. **snapshot 系统**

P0 实现后,工作台从"列图 + 打标"升到"能完成一次完整数据集制备"。

P1 再补:
- Anima Tagger 集成
- 美学打分模型(Schuhmann predictor)
- 自动旋转 EXIF / 去黑边
- 跨数据集复制(Intake from-dataset)
- caption 编辑器自动补全

---

## 7. 工作量评估

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **F0** | UI 主结构(stepper + history bar) | 1 天 |
| **F1** | Audit 全维度扫描 + 报告 + 直方图 | 2-3 天 |
| **F2** | Curate 智能裁剪 + 隔离区 + 重采样 + 黑名单批删 | 3-4 天 |
| **F3** | Annotate 标签词表 + find-replace + 触发词管理 + diff | 2-3 天 |
| **F4** | Ship lint + 真导出 zip + 训练任务联动 | 2 天 |
| **F5** | Snapshot 系统 + ops 升级(逆向 op) | 2 天 |
| **F6** | similarity 接前端 + Anima Tagger 接前端 | 1 天 |

合计 13-16 天。这才是"真正能做数据集处理的工作台"应付的工作量;之前的 5.5 天 tabs 重构只是化妆。

---

## 8. 不在范围

- **生成式数据增强**(SD 加强 / 风格迁移加噪) — 训练数据集不该走这条
- **手画 mask / 分割编辑** — canvas 工程量级,独立产品
- **多用户协作 / 评论 / 审批** — 单机自托管定位
