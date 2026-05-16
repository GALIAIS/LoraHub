import { AlertTriangle, Gauge } from "lucide-react"
import { api } from "@/lib/api"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function PreflightPanel({
  preflight,
}: {
  preflight: NonNullable<Awaited<ReturnType<typeof api.validateConfig>>["preflight"]>
}) {
  const warnings = preflight.issues.filter((issue) => issue.severity !== "info")
  const missingCaptions = preflight.paths.missing_caption_files

  return (
    <Card className="mx-4 mt-3 rounded-[6px] border-border/60 bg-card/80 shadow-[var(--panel-shadow)]">
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              起飞前自检
            </div>
            <div className="mt-1 text-sm font-medium">
              {warnings.length === 0 ? "可以启动训练" : `还有 ${warnings.length} 项需要关注`}
            </div>
          </div>
          <div className="rounded-[4px] border border-border/70 px-3 py-2 text-right">
            <div className="flex items-center justify-end gap-1.5 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              <Gauge className="size-3" /> 显存预估
            </div>
            <div className="font-mono text-lg leading-none mt-1">
              {preflight.vram.total_gib.toFixed(2)} GiB
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-3">
          <PreflightMetric
            label="模型"
            value={preflight.paths.checkpoint_exists ? "已就绪" : "缺失"}
            ok={preflight.paths.checkpoint_exists}
          />
          <PreflightMetric
            label="数据集"
            value={`${preflight.paths.image_files} 张图片`}
            ok={preflight.paths.dataset_exists && preflight.paths.image_files > 0}
          />
          <PreflightMetric
            label="标注"
            value={`${preflight.paths.caption_files}/${preflight.paths.image_files}`}
            ok={missingCaptions.length === 0}
          />
        </div>

        {warnings.length > 0 && (
          <ul className="mt-3 space-y-1.5 text-xs">
            {warnings.slice(0, 5).map((issue, i) => (
              <li key={i} className="flex items-start gap-2 text-muted-foreground">
                <AlertTriangle className="mt-0.5 size-3 text-amber-600 dark:text-amber-400" />
                <span>
                  <span className="font-mono text-foreground">{issue.field}</span>: {issue.message}
                </span>
              </li>
            ))}
          </ul>
        )}

        {missingCaptions.length > 0 && (
          <div className="mt-3 rounded-[4px] bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            缺失标注：
            <span className="font-mono text-foreground">
              {missingCaptions.join(", ")}
              {preflight.paths.missing_caption_files_truncated ? "，…" : ""}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function PreflightMetric({
  label,
  value,
  ok,
}: {
  label: string
  value: string
  ok: boolean
}) {
  return (
    <div className="rounded-[4px] border border-border/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={cn("mt-1 text-sm font-medium", ok ? "text-emerald-600 dark:text-emerald-400" : "text-amber-700 dark:text-amber-400")}>
        {value}
      </div>
    </div>
  )
}
