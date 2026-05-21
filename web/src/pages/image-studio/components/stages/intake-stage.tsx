import { ArrowRight } from "lucide-react"

interface Props {
  datasetPath: string
}

export function IntakeStage({ datasetPath }: Props) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center text-sm text-muted-foreground">
        <p className="mb-2 text-foreground">📥 导入 Stage(规划中)</p>
        <p>当前数据集: <code className="text-xs">{datasetPath}</code></p>
        <p className="mt-3">
          导入向导(本地路径吸附 / 跨数据集复制 / phash 预去重 /
          EXIF 自动旋转)将在 P1 阶段落地。
        </p>
        <p className="mt-3 inline-flex items-center gap-1">
          目前请直接在 整理 Stage 拖入文件 <ArrowRight className="size-3" />
        </p>
      </div>
    </div>
  )
}
