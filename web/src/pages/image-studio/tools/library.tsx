/**
 * 工具库 — 直接复用既有 library 面板（panel 已经是独立组件）。
 */
import { TagLibraryPanel } from "../components/library/tag-library-panel"
import { TriggerLibraryPanel } from "../components/library/trigger-library-panel"
import { PromptLibraryPanel } from "../components/library/prompt-library-panel"

export function LibraryTagsTool() {
  return <TagLibraryPanel />
}

export function LibraryTriggersTool() {
  return <TriggerLibraryPanel />
}

export function LibraryPromptsTool() {
  return <PromptLibraryPanel />
}
