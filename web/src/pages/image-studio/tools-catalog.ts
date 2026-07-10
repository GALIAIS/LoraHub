/**
 * 图像工作台"全部工具"广场的元数据。
 *
 * 后端 13 个 image-studio 子路由对应的工具被分到 5 个类别下，每条工具能直接
 * 跳到对应的 stage 子页（带 ?tool=<id> URL 标识，stage 子页可选地用此参数
 * 高亮 / 滚动到对应面板）。stepper 不再是必经之路 — 用户既可走阶段路径，也
 * 能从工具广场点任意工具直达。
 *
 * 没有为每个工具单独渲染对话框，因为同 stage 子页内已有该工具的现成面板。
 * 复用现成 UI 即可，避免代码重复。
 */
import type { LucideIcon } from "lucide-react"
import {
  AlertTriangle,
  Boxes,
  ClipboardList,
  Copy,
  Eraser,
  FileSearch,
  FilterX,
  FolderInput,
  Gauge,
  ImagePlus,
  Layers,
  Library,
  PackageCheck,
  RefreshCw,
  Rocket,
  Replace,
  RotateCcw,
  Save,
  ScanSearch,
  Scissors,
  ShieldQuestion,
  Sparkles,
  Tags,
  Trash,
  TriangleAlert,
  Wand2,
  Wrench,
} from "lucide-react"
import type { StageId } from "./components/stage-stepper"

export type ToolCategory =
  | "intake"
  | "audit"
  | "curate"
  | "tagging"
  | "ai"
  | "ship"
  | "library"

export interface ToolCategoryInfo {
  id: ToolCategory
  label: string
  description: string
  icon: LucideIcon
}

export const TOOL_CATEGORIES: readonly ToolCategoryInfo[] = [
  { id: "intake",  label: "导入",   description: "把图片接到工作台，支持本地路径吸附与跨数据集挑图",      icon: FolderInput   },
  { id: "audit",   label: "审计",   description: "扫尺寸 / 比例 / 重复 / 相似度 / 触发词覆盖",            icon: ClipboardList },
  { id: "curate",  label: "整理",   description: "EXIF 校正、批量缩放、隔离区、备份回滚",                  icon: Scissors      },
  { id: "tagging", label: "打标",   description: "WD14 / JoyTag 自动打标 + 标签词频与批量编辑",            icon: Tags          },
  { id: "ai",      label: "AI",     description: "VLM 直出 caption / 质量分 / 触发词 / Anima 重写",         icon: Sparkles      },
  { id: "ship",    label: "出口",   description: "训练就绪门禁、导出复制、另存数据集",                      icon: PackageCheck  },
  { id: "library", label: "工具库", description: "跨数据集的标签词典、触发词索引",                            icon: Library       },
] as const

export interface ToolInfo {
  /** 在 URL 里出现的稳定 id (?tool=<id>)。也作为 stage 子页面板高亮的 hint。 */
  id: string
  category: ToolCategory
  /** 工具进入哪个 stage 子页 — 即点击卡片后路由的 ?stage 参数。 */
  stage: StageId | "library" | "lora-test"
  label: string
  /** 一句话描述 — 1-2 句，控制在 60 中文字符内便于卡片排版。 */
  description: string
  icon: LucideIcon
  /** 是否需要先选中数据集；广场页据此禁用 / 提示。 */
  requiresDataset: boolean
  /** 是否走异步会话（前端会进入轮询 / 任务条），仅用于卡片角标。 */
  async?: boolean
  /** 是否会写入文件（卡片右下角小标，用于提醒"自动备份"）。 */
  writes?: boolean
}

/**
 * 全部工具清单。新增后端工具时往这里追加；id 一旦发布不要改（URL 里写过）。
 */
export const TOOLS: readonly ToolInfo[] = [
  // ===== 导入 =====
  {
    id: "intake-preflight",
    category: "intake",
    stage: "intake",
    label: "源目录预检",
    description: "扫源目录给报告，列出图片数 / 重名 / 已存在 sidecar",
    icon: ScanSearch,
    requiresDataset: true,
  },
  {
    id: "intake-local-path",
    category: "intake",
    stage: "intake",
    label: "本地路径吸附",
    description: "把磁盘任意目录里的图片导入当前数据集，保留 .txt sidecar",
    icon: ImagePlus,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "intake-from-dataset",
    category: "intake",
    stage: "intake",
    label: "跨数据集挑图",
    description: "从已存在数据集挑一部分图进新数据集，避免覆盖",
    icon: Copy,
    requiresDataset: true,
    writes: true,
  },

  // ===== 审计 =====
  {
    id: "audit-scan",
    category: "audit",
    stage: "audit",
    label: "数据集体检",
    description: "尺寸 / 比例 / 重复 caption / 触发词命中率 — cheap-pass 不耗 GPU",
    icon: Gauge,
    requiresDataset: true,
  },
  {
    id: "dedupe-l1",
    category: "audit",
    stage: "audit",
    label: "L1 感知哈希查重",
    description: "phash 聚类找像素级重复 / 近重复，支持批删",
    icon: Layers,
    requiresDataset: true,
  },
  {
    id: "similarity-l2",
    category: "audit",
    stage: "audit",
    label: "L2 语义相似",
    description: "AI 嵌入找语义重复（同角度 / 同主体不同噪声），耗 GPU",
    icon: Boxes,
    requiresDataset: true,
    async: true,
  },

  // ===== 整理 =====
  {
    id: "curate-overview",
    category: "curate",
    stage: "curate",
    label: "整理总览",
    description: "网格 / inspector / 上传入口的一站式视图，单图操作都在这里",
    icon: Layers,
    requiresDataset: true,
  },
  {
    id: "curate-auto-rotate",
    category: "curate",
    stage: "curate",
    label: "EXIF 自动旋转",
    description: "按 EXIF orientation 校正纵向 / 横向，无元数据则跳过",
    icon: RefreshCw,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "curate-batch-resize",
    category: "curate",
    stage: "curate",
    label: "批量缩放",
    description: "按目标边长 / 长宽比批量重采样，自动备份原图",
    icon: Wrench,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "curate-quarantine",
    category: "curate",
    stage: "curate",
    label: "隔离区",
    description: "把疑似图先丢隔离区不删，可一键还原",
    icon: ShieldQuestion,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "curate-restore-backup",
    category: "curate",
    stage: "curate",
    label: "备份回滚",
    description: "整理 / caption 写操作每次都备份，本工具列出可还原版本",
    icon: RotateCcw,
    requiresDataset: true,
  },

  // ===== 打标 =====
  {
    id: "tagging-wd14",
    category: "tagging",
    stage: "annotate",
    label: "WD14 / JoyTag 打标",
    description: "本地 ONNX / PyTorch 模型给图片自动出 booru 风格标签",
    icon: Wand2,
    requiresDataset: true,
    async: true,
    writes: true,
  },
  {
    id: "captions-vocab",
    category: "tagging",
    stage: "annotate",
    label: "标签词频",
    description: "统计当前数据集的 tag 分布，定位错标 / 低频 / 拼写问题",
    icon: FileSearch,
    requiresDataset: true,
  },
  {
    id: "captions-find-replace",
    category: "tagging",
    stage: "annotate",
    label: "批量找替换",
    description: "对所有 caption 文件批量替换 tag，自动备份原文件",
    icon: Replace,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "captions-inject-trigger",
    category: "tagging",
    stage: "annotate",
    label: "注入触发词",
    description: "在所有 caption 头部按规则注入触发词 / 角色名",
    icon: Tags,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "captions-blacklist",
    category: "tagging",
    stage: "annotate",
    label: "标签黑名单",
    description: "按列表批量删除 tag，删前自动备份",
    icon: FilterX,
    requiresDataset: true,
    writes: true,
  },

  // ===== AI =====
  {
    id: "lora-testbench",
    category: "ai",
    stage: "lora-test",
    label: "LoRA 生图测试",
    description: "选择训练出的 LoRA checkpoint，调 prompt / seed / CFG 直接验收效果",
    icon: Rocket,
    requiresDataset: false,
    async: true,
    writes: true,
  },
  {
    id: "ai-caption",
    category: "ai",
    stage: "annotate",
    label: "VLM 批量 caption",
    description: "外部 VLM 直接给图片写 caption（一步），适合通用风格数据集",
    icon: Sparkles,
    requiresDataset: true,
    async: true,
    writes: true,
  },
  {
    id: "ai-smart-caption",
    category: "ai",
    stage: "annotate",
    label: "智能 caption",
    description: "WD14 出标签 → VLM 据此合成 Anima caption（两步组合）",
    icon: Wand2,
    requiresDataset: true,
    async: true,
    writes: true,
  },
  {
    id: "ai-wd14-prefilter",
    category: "ai",
    stage: "annotate",
    label: "WD14 单步出标签",
    description: "智能 caption 的第一步独立运行：WD14 + prompt 组装，不调 VLM",
    icon: Tags,
    requiresDataset: true,
  },
  {
    id: "ai-vlm-anima-rewrite",
    category: "ai",
    stage: "annotate",
    label: "VLM Anima 重写",
    description: "智能 caption 的第二步独立运行：用现成标签调 VLM 写 Anima caption",
    icon: Sparkles,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "ai-quality",
    category: "ai",
    stage: "annotate",
    label: "AI 质量评分",
    description: "对每张图打 0-10 质量分 + 标签 + 原因，用于筛除低质",
    icon: Gauge,
    requiresDataset: true,
    async: true,
  },
  {
    id: "ai-trigger-words",
    category: "ai",
    stage: "annotate",
    label: "AI 触发词抽取",
    description: "让 VLM 从图片 / caption 里生成触发词候选",
    icon: TriangleAlert,
    requiresDataset: true,
    async: true,
  },

  // ===== 出口 =====
  {
    id: "ship-lint",
    category: "ship",
    stage: "ship",
    label: "训练就绪门禁",
    description: "检查 caption 完整性 / 数量 / 触发词覆盖；通过才允许开训",
    icon: AlertTriangle,
    requiresDataset: true,
  },
  {
    id: "ship-export",
    category: "ship",
    stage: "ship",
    label: "导出复制",
    description: "把数据集导出到指定目录（kohya / Anima 标准格式）",
    icon: Save,
    requiresDataset: true,
    writes: true,
  },
  {
    id: "ship-save-as",
    category: "ship",
    stage: "ship",
    label: "另存数据集",
    description: "把当前数据集另存为新数据集副本，原集保留不动",
    icon: PackageCheck,
    requiresDataset: true,
    writes: true,
  },

  // ===== 工具库（跨数据集，不需要选数据集） =====
  {
    id: "library-tags",
    category: "library",
    stage: "library",
    label: "标签词典",
    description: "维护全局 tag + 别名 + 分类 + 颜色，跨数据集复用",
    icon: Tags,
    requiresDataset: false,
  },
  {
    id: "library-triggers",
    category: "library",
    stage: "library",
    label: "触发词索引",
    description: "trigger word ↔ 角色 / 概念 ↔ 数据集映射",
    icon: Wand2,
    requiresDataset: false,
  },
] as const

// 上面 import 列表里有几个未被实际使用的 icon（保留以便后续阶段 2/3 新增工具时取用）。
// 显式标记一次避免 lint 噪音。
void Eraser
void Trash
