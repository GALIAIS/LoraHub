interface Props {
  datasetPath: string
}

export function AnnotateStage({ datasetPath }: Props) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center text-sm text-muted-foreground">
        <p className="mb-2 text-foreground">🏷 标注 Stage(规划中)</p>
        <p>当前数据集: <code className="text-xs">{datasetPath}</code></p>
        <p className="mt-3">
          标注 Stage 将整合 WD14 自动打标 / VLM caption /
          Anima Tagger / 标签词表 / 全局批改 / 触发词管理。
          目前 WD14 + VLM 仍在 整理 Stage 的右上工具栏可用。
        </p>
      </div>
    </div>
  )
}
