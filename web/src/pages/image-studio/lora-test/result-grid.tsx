/**
 * 结果网格 — 展示生成图片缩略图与 XY 拼接图。
 */
import { ImageIcon } from "lucide-react"
import { api } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { formatAxisLabel } from "./helpers"
import type { ResultImage } from "./types"

export function ResultGrid({
  sessionId,
  images,
  gridPath,
  loading,
  onOpen,
}: {
  sessionId: string
  images: ResultImage[]
  gridPath: string | null
  loading: boolean
  onOpen: (image: ResultImage) => void
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>结果</CardTitle>
        <CardDescription>固定比例网格展示，参数写入 sidecar 可追溯。</CardDescription>
      </CardHeader>
      <CardContent>
        {loading && <Skeleton className="h-64" />}
        {!loading && images.length === 0 && (
          <div className="grid min-h-64 place-items-center rounded-[6px] border border-dashed border-border/70 bg-muted/25 text-center">
            <div>
              <ImageIcon className="mx-auto size-8 text-muted-foreground" />
              <p className="mt-2 text-sm font-medium">还没有生成结果</p>
              <p className="mt-1 text-xs text-muted-foreground">
                生成完成后会显示图片、seed 和核心参数。
              </p>
            </div>
          </div>
        )}
        {gridPath && (
          <button
            type="button"
            className="mb-3 block w-full overflow-hidden rounded-[6px] border border-border/60 bg-background text-left"
            onClick={() =>
              window.open(
                api.loraTestResultFileUrl(sessionId, gridPath),
                "_blank",
                "noopener,noreferrer",
              )
            }
          >
            <img
              src={api.loraTestResultFileUrl(sessionId, gridPath)}
              alt="XY grid"
              className="max-h-[70vh] w-full object-contain"
            />
          </button>
        )}
        {images.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {images.map((image) => (
              <button
                key={image.path}
                type="button"
                className="group overflow-hidden rounded-[6px] border border-border/60 bg-muted/20 text-left"
                onClick={() => onOpen(image)}
              >
                <div className="grid aspect-[9/13] place-items-center bg-background">
                  <img
                    src={api.loraTestResultFileUrl(sessionId, image.path)}
                    alt={`seed ${image.seed}`}
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
                <div className="flex items-center justify-between gap-2 px-2 py-2 text-[11px]">
                  <span className="font-mono">seed {image.seed}</span>
                  <span className="text-muted-foreground">
                    {formatAxisLabel(image) ?? `${image.steps} / cfg ${image.cfg}`}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
