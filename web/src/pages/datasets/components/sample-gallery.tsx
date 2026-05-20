import { useState } from "react"
import { ImageOff, Pencil } from "lucide-react"
import { api, type DatasetScanResponse } from "@/lib/api"
import { Badge } from "@/components/ui/badge"

type Sample = DatasetScanResponse["samples"][number]

const IMAGE_EXTS = new Set(["bmp", "gif", "jpeg", "jpg", "png", "webp"])

function isImageSample(sample: Sample): boolean {
  const ext = sample.name.split(".").pop()?.toLowerCase()
  return !!ext && IMAGE_EXTS.has(ext)
}

export function SampleGallery({
  samples,
  onEdit,
  onPreviewImage,
}: {
  samples: Sample[]
  onEdit: (sample: Sample) => void
  onPreviewImage?: (sample: Sample) => void
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
      {samples.map((sample) => (
        <SampleCard
          key={sample.relative_path}
          sample={sample}
          onEdit={() => onEdit(sample)}
          onPreviewImage={onPreviewImage ? () => onPreviewImage(sample) : undefined}
        />
      ))}
    </div>
  )
}

function SampleCard({
  sample,
  onEdit,
  onPreviewImage,
}: {
  sample: Sample
  onEdit: () => void
  onPreviewImage?: () => void
}) {
  const isImage = isImageSample(sample)
  const [broken, setBroken] = useState(false)
  return (
    <div className="relative rounded-[6px] border border-border/60 bg-card/60 overflow-hidden transition-colors hover:border-primary/40 flex flex-col">
      <button
        type="button"
        onClick={onPreviewImage ?? onEdit}
        className="group block bg-muted/40 w-full"
        title={sample.caption ?? sample.name}
      >
        {isImage ? (
          <div className="aspect-square overflow-hidden">
            {broken ? (
              <div className="w-full h-full grid place-items-center text-muted-foreground/70 text-[11px] gap-1 flex-col flex">
                <ImageOff className="size-5" />
                <span>缩略图不可用</span>
              </div>
            ) : (
              <img
                src={api.datasetThumbUrl(sample.path, 336)}
                loading="lazy"
                alt={sample.name}
                className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform"
                onError={() => setBroken(true)}
              />
            )}
          </div>
        ) : (
          <div className="aspect-square grid place-items-center text-[11px] text-muted-foreground/70 font-mono px-3 text-center">
            {sample.name}
          </div>
        )}
      </button>
      <div className="flex-1 min-w-0 px-2 py-2 space-y-1.5">
        <div className="flex items-center gap-2">
          <div
            className="font-mono text-[11px] truncate flex-1"
            title={sample.relative_path}
          >
            {sample.relative_path}
          </div>
          <Badge
            variant={sample.caption_exists ? "secondary" : "outline"}
            className="rounded-[2px] text-[10px] py-0 px-1.5"
          >
            {sample.caption_exists ? "已标注" : "缺 .txt"}
          </Badge>
        </div>
        <div className="flex items-start gap-2">
          <p className="text-[11px] text-muted-foreground line-clamp-2 flex-1">
            {sample.caption ?? "暂无标注"}
          </p>
          <button
            type="button"
            onClick={onEdit}
            className="text-[11px] text-primary hover:underline shrink-0 inline-flex items-center gap-1"
          >
            <Pencil className="size-3" /> 编辑
          </button>
        </div>
      </div>
    </div>
  )
}
