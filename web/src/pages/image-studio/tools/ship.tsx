/**
 * 出口类工具 — 训练就绪 / 导出 / 另存。直接复用 ship-stage 里
 * 已经抽出来的三个 panel，每个 panel 自包含。
 */
import {
  ShipLintCard,
  ExportPanel as ShipExportPanelImpl,
  SaveAsPanel as ShipSaveAsPanelImpl,
} from "../components/stages/ship-stage"

export function ShipLintTool({ datasetPath }: { datasetPath: string }) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <ShipLintCard datasetPath={datasetPath} />
    </div>
  )
}

export function ShipExportTool({ datasetPath }: { datasetPath: string }) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <ShipExportPanelImpl datasetPath={datasetPath} />
    </div>
  )
}

export function ShipSaveAsTool({ datasetPath }: { datasetPath: string }) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <ShipSaveAsPanelImpl datasetPath={datasetPath} />
    </div>
  )
}
