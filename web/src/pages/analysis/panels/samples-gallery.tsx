/**
 * SamplesGallery — virtualised-friendly grid of sample-image artifacts
 * grouped by epoch / step (parsed from the filename), with a lightbox
 * for full-size inspection.
 */
import { useMemo, useState } from "react"
import { ImageIcon, X } from "lucide-react"
import { api, type JobFile } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface SampleEntry extends JobFile {
  epoch: number | null
  step: number | null
}

function parseSampleMeta(path: string): {
  epoch: number | null
  step: number | null
} {
  const name = path.split(/[\\/]/).pop() ?? ""
  const epochMatch = name.match(/e(?:poch)?[-_]?(\d+)/i)
  const stepMatch = name.match(/s(?:tep)?[-_]?(\d+)/i)
  return {
    epoch: epochMatch ? Number(epochMatch[1]) : null,
    step: stepMatch ? Number(stepMatch[1]) : null,
  }
}

export function SamplesGallery({
  jobId,
  samples,
  loading,
}: {
  jobId: string
  samples: JobFile[]
  loading: boolean
}) {
  const [openSrc, setOpenSrc] = useState<string | null>(null)

  const enriched = useMemo<SampleEntry[]>(() => {
    return samples
      .map((s) => ({ ...s, ...parseSampleMeta(s.path) }))
      .sort((a, b) => {
        const ae = a.epoch ?? Number.MAX_SAFE_INTEGER
        const be = b.epoch ?? Number.MAX_SAFE_INTEGER
        if (ae !== be) return ae - be
        const as = a.step ?? Number.MAX_SAFE_INTEGER
        const bs = b.step ?? Number.MAX_SAFE_INTEGER
        if (as !== bs) return as - bs
        return a.modified_at - b.modified_at
      })
  }, [samples])

  return (
    <Card className="rounded-[6px] border-border/60">
      <CardHeader className="py-2.5 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          样本预览画廊
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading ? "加载中…" : `${enriched.length} 张样本`}
        </span>
      </CardHeader>
      <CardContent className="p-4">
        {enriched.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground">
            <ImageIcon className="size-8 opacity-40" />
            <span className="text-xs">
              尚未生成样本图。配置 sampling.everyNEpochs 后会自动出现
            </span>
          </div>
        )}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
          {enriched.map((s) => {
            const url = api.jobFileUrl(jobId, s.path)
            const name = s.path.split(/[\\/]/).pop() ?? s.path
            return (
              <button
                key={s.path}
                type="button"
                onClick={() => setOpenSrc(url)}
                className="group relative aspect-square overflow-hidden rounded-[4px] border border-border/60 bg-muted/20 transition hover:border-primary/60"
                title={name}
              >
                <img
                  src={url}
                  alt={name}
                  loading="lazy"
                  className="h-full w-full object-cover transition group-hover:scale-105"
                />
                <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 text-[10px] font-medium text-white opacity-0 transition group-hover:opacity-100">
                  {s.epoch != null && <span className="mr-2">e{s.epoch}</span>}
                  {s.step != null && <span>s{s.step}</span>}
                  {s.epoch == null && s.step == null && (
                    <span className="truncate">{name}</span>
                  )}
                </div>
              </button>
            )
          })}
        </div>
      </CardContent>
      {openSrc && <Lightbox src={openSrc} onClose={() => setOpenSrc(null)} />}
    </Card>
  )
}

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
      onClick={onClose}
    >
      <Button
        variant="ghost"
        size="sm"
        className="absolute top-4 right-4 text-white hover:bg-white/10"
        onClick={onClose}
      >
        <X className="size-4" />
      </Button>
      <img
        src={src}
        alt="sample preview"
        className="max-h-[90vh] max-w-[90vw] rounded-[4px] object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  )
}
