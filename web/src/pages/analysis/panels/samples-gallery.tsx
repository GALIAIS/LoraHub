/**
 * SamplesGallery — virtualised-friendly grid of sample-image artifacts
 * grouped by epoch / step (parsed from the path).
 */
import { useMemo } from "react"
import { ImageIcon } from "lucide-react"
import { api, type JobFile } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface SampleEntry extends JobFile {
  epoch: number | null
  step: number | null
}

function parseSampleMeta(path: string): {
  epoch: number | null
  step: number | null
} {
  const normalized = path.replace(/\\/g, "/")
  const name = normalized.split("/").pop() ?? ""
  const compact = name.match(/^(?<epoch>\d+)[_-](?<step>\d{3,})(?:[_-]|$)/)
  const epochMatch =
    normalized.match(/(?:^|[/_-])e(?:poch)?[-_]?(\d+)(?:[/_.-]|$)/i) ??
    normalized.match(/(?:^|[/_-])epoch[-_]?(\d+)(?:[/_.-]|$)/i)
  const stepMatch =
    normalized.match(/(?:^|[/_-])s(?:tep)?[-_]?(\d+)(?:[/_.-]|$)/i) ??
    normalized.match(/(?:^|[/_-])step[-_]?(\d+)(?:[/_.-]|$)/i) ??
    name.match(/(?:^|_)(\d{3,})_\d+_\d{10,14}(?:_|\.|$)/)
  return {
    epoch: compact?.groups?.epoch
      ? Number(compact.groups.epoch)
      : epochMatch
        ? Number(epochMatch[1])
        : null,
    step: compact?.groups?.step
      ? Number(compact.groups.step)
      : stepMatch
        ? Number(stepMatch[1])
        : null,
  }
}

function openExternal(url: string): void {
  window.open(url, "_blank", "noopener,noreferrer")
}

export function SamplesGallery({
  jobId,
  samples,
  loading,
  triggerWord,
}: {
  jobId: string
  samples: JobFile[]
  loading: boolean
  /** 触发词，作为 LoRA 预览图的橙色角标文字。null = 显示 LORA。*/
  triggerWord?: string | null
}) {
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
    <Card>
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
        {/* Local scroll container so a long sample run (hundreds of
            checkpoints × multiple prompts) doesn't push the loss chart
            and KPIs off-screen. Cap at ~5 rows of the densest grid
            (lg = 5 cols × 5 rows ≈ 70vh). The aspect-square children
            keep predictable thumbnail sizes inside the scroller. */}
        <div className="max-h-[70vh] overflow-y-auto pr-1 -mr-1">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {enriched.map((s) => {
              const url = api.jobFileUrl(jobId, s.path)
              const name = s.path.split(/[\\/]/).pop() ?? s.path
              const isBaseline = s.step === 0 || (s.step == null && s.epoch === 0)
              // 触发词 ASCII 化作为左上角橙色角标文字，超长截断；
              // 缺失时退回到字面 LORA。
              const loraBadge = (triggerWord || "").trim() || "LORA"
              const loraBadgeShort =
                loraBadge.length > 18 ? loraBadge.slice(0, 17) + "…" : loraBadge
              return (
                <button
                  key={s.path}
                  type="button"
                  onClick={() => openExternal(url)}
                  className={cn(
                    "group relative aspect-square overflow-hidden rounded-[4px] border bg-muted/20 transition",
                    isBaseline
                      ? "border-sky-400/70 ring-1 ring-sky-400/30 hover:border-sky-500"
                      : "border-amber-400/60 hover:border-amber-500/80",
                  )}
                  title={
                    isBaseline
                      ? `[基模] ${name}`
                      : `[${loraBadge}] ${name}`
                  }
                >
                  <img
                    src={url}
                    alt={name}
                    loading="lazy"
                    className="h-full w-full object-cover transition group-hover:scale-105"
                  />
                  {isBaseline ? (
                    <div className="pointer-events-none absolute inset-x-0 top-0 bg-sky-500/80 px-1 py-px text-center text-[9px] font-medium text-white leading-tight">
                      BASE
                    </div>
                  ) : (
                    <div
                      className="pointer-events-none absolute left-0 top-0 max-w-[80%] truncate bg-amber-500/85 px-1.5 py-px text-[9px] font-medium uppercase tracking-wide text-white leading-tight rounded-br-[3px] shadow-sm"
                      title={loraBadge}
                    >
                      {loraBadgeShort}
                    </div>
                  )}
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
        </div>
      </CardContent>
    </Card>
  )
}
