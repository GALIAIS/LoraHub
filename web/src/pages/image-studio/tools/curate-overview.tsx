/**
 * 整理总览 — 复用既有 DatasetDetail（带网格 / inspector / 上传 / AI 批量入口）。
 *
 * DatasetDetail 自己从 URL 读 path / page / view / sort，所以容器只需要原样塞进
 * 来即可；datasetPath 由 tool-page 下发，但 DatasetDetail 仍走 URL 拿值，避免
 * 双源同步出错。
 */
import { DatasetDetail } from "../components/dataset-detail"

export function CurateOverviewTool({ datasetPath }: { datasetPath: string }) {
  // datasetPath 已经在 URL 里（tool-page 把它放在 ?path= 上），DatasetDetail
  // 内部用 useSearchParams 自取，因此这里无需再传。引用一下避免 lint。
  void datasetPath
  return (
    <div className="h-full overflow-hidden">
      <DatasetDetail />
    </div>
  )
}
