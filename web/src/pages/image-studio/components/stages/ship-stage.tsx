interface Props {
  datasetPath: string
}

export function ShipStage({ datasetPath }: Props) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center text-sm text-muted-foreground">
        <p className="mb-2 text-foreground">🚀 输出 Stage(规划中)</p>
        <p>当前数据集: <code className="text-xs">{datasetPath}</code></p>
        <p className="mt-3">
          训练前 lint / 真导出 zip / 同步到 VPS / 一键新建训练任务。
          P4 落地。
        </p>
      </div>
    </div>
  )
}
