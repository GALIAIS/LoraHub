/**
 * tool id -> Body 组件 的映射。
 *
 * 工具 panel 都期望一个统一的 props 形态 `{ datasetPath: string }`,
 * tool-page.tsx 在 requiresDataset=true 时已保证 datasetPath 非空,
 * 所以 panel 内部不需要再做 datasetPath 兜底。
 *
 * 暂时未填映射的工具，tool-page 会渲染"即将上线"占位。
 */
import type { ComponentType } from "react"
import {
  IntakePreflightTool,
  IntakeLocalPathTool,
  IntakeFromDatasetTool,
} from "./tools/intake"
import {
  AuditScanTool,
  DedupeL1Tool,
  SimilarityL2Tool,
} from "./tools/audit"
import {
  CurateAutoRotateTool,
  CurateBatchResizeTool,
  CurateQuarantineTool,
  CurateRestoreBackupTool,
} from "./tools/curate"
import {
  CaptionsVocabTool,
  CaptionsFindReplaceTool,
  CaptionsInjectTriggerTool,
  CaptionsBlacklistTool,
  TaggingWd14Tool,
} from "./tools/tagging"
import {
  AiCaptionTool,
  AiSmartCaptionTool,
  AiWd14PrefilterTool,
  AiVlmAnimaRewriteTool,
  AiQualityTool,
  AiTriggerWordsTool,
} from "./tools/ai"
import { ShipLintTool, ShipExportTool, ShipSaveAsTool } from "./tools/ship"
import {
  LibraryTagsTool,
  LibraryTriggersTool,
  LibraryPromptsTool,
} from "./tools/library"
import { CurateOverviewTool } from "./tools/curate-overview"

export type ToolBodyProps = { datasetPath: string }

export const TOOL_COMPONENTS: Record<string, ComponentType<ToolBodyProps>> = {
  // intake
  "intake-preflight": IntakePreflightTool,
  "intake-local-path": IntakeLocalPathTool,
  "intake-from-dataset": IntakeFromDatasetTool,
  // audit
  "audit-scan": AuditScanTool,
  "dedupe-l1": DedupeL1Tool,
  "similarity-l2": SimilarityL2Tool,
  // curate
  "curate-overview": CurateOverviewTool,
  "curate-auto-rotate": CurateAutoRotateTool,
  "curate-batch-resize": CurateBatchResizeTool,
  "curate-quarantine": CurateQuarantineTool,
  "curate-restore-backup": CurateRestoreBackupTool,
  // tagging
  "tagging-wd14": TaggingWd14Tool,
  "captions-vocab": CaptionsVocabTool,
  "captions-find-replace": CaptionsFindReplaceTool,
  "captions-inject-trigger": CaptionsInjectTriggerTool,
  "captions-blacklist": CaptionsBlacklistTool,
  // ai
  "ai-caption": AiCaptionTool,
  "ai-smart-caption": AiSmartCaptionTool,
  "ai-wd14-prefilter": AiWd14PrefilterTool,
  "ai-vlm-anima-rewrite": AiVlmAnimaRewriteTool,
  "ai-quality": AiQualityTool,
  "ai-trigger-words": AiTriggerWordsTool,
  // ship
  "ship-lint": ShipLintTool,
  "ship-export": ShipExportTool,
  "ship-save-as": ShipSaveAsTool,
  // library
  "library-tags": LibraryTagsTool,
  "library-triggers": LibraryTriggersTool,
  "library-prompts": LibraryPromptsTool,
}
