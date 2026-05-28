/**
 * 导入类工具 — 服务器路径导入 / 跨数据集复制 / 预扫。
 *
 * "预扫"工具页直接复用 LocalPathPanel（预扫和导入流程上紧耦合，
 * 让用户在同一个面板上预扫 -> 导入是更顺的体验）。
 */
import { ArrowRight } from "lucide-react"
import { Link } from "react-router-dom"
import {
  FromDatasetPanel,
  LocalPathPanel,
} from "../components/stages/intake-stage"

export function IntakeLocalPathTool({ datasetPath }: { datasetPath: string }) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <LocalPathPanel datasetPath={datasetPath} />
    </div>
  )
}

export function IntakeFromDatasetTool({
  datasetPath,
}: {
  datasetPath: string
}) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <FromDatasetPanel datasetPath={datasetPath} />
    </div>
  )
}

export function IntakePreflightTool({
  datasetPath,
}: {
  datasetPath: string
}) {
  // Preflight is the "scan first" half of the local-path workflow.
  // Rendering the same panel keeps the UX coherent — one form, one
  // "预扫" button, one "导入" button — instead of forcing the user to
  // hop between two pages just to validate then import.
  const search = new URLSearchParams({ path: datasetPath }).toString()
  return (
    <div className="h-full overflow-y-auto p-4 space-y-3">
      <div className="rounded-md border bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground flex items-center gap-2">
        <span>
          预扫和导入共用同一个面板 — 先点
          <span className="font-medium">预扫</span>
          看候选 / 重复统计，再点
          <span className="font-medium">导入</span>
          实际拷文件。
        </span>
        <Link
          to={`/image-studio/tools/intake-local-path?${search}`}
          className="ml-auto inline-flex items-center gap-1 text-foreground hover:underline"
        >
          直接去导入
          <ArrowRight className="size-3" />
        </Link>
      </div>
      <LocalPathPanel datasetPath={datasetPath} />
    </div>
  )
}
