/**
 * StageTimeline — horizontal bar visualising the PELT-derived training
 * segments. Each segment is rendered proportional to its step span and
 * coloured by stage; hovering reveals slope + mean loss.
 *
 * The component is purely presentational — segmentation is computed by
 * the parent (or by ``analyseChangepoints`` from ``pelt.ts``). This
 * keeps the same analysis available to other consumers (e.g. PR diff
 * comparisons) without coupling them to a UI shape.
 */
import type { PeltSegment } from "./pelt"
import { cn } from "@/lib/utils"

const STAGE_LABEL: Record<PeltSegment["stage"], string> = {
  warmup: "热身",
  converging: "收敛",
  plateau: "平台",
  diverging: "发散",
}

const STAGE_COLOR: Record<PeltSegment["stage"], string> = {
  warmup: "bg-sky-500/65",
  converging: "bg-emerald-500/65",
  plateau: "bg-amber-500/55",
  diverging: "bg-red-500/70",
}

const STAGE_TEXT: Record<PeltSegment["stage"], string> = {
  warmup: "text-sky-700 dark:text-sky-300",
  converging: "text-emerald-700 dark:text-emerald-300",
  plateau: "text-amber-700 dark:text-amber-300",
  diverging: "text-red-600 dark:text-red-400",
}

export function StageTimeline({
  segments,
  changepointSteps,
}: {
  segments: PeltSegment[]
  changepointSteps: number[]
}) {
  if (segments.length === 0) return null
  const totalSpan =
    segments[segments.length - 1].endStep - segments[0].startStep || 1

  return (
    <div className="rounded-[6px] border border-border/60 bg-card/60 px-3.5 py-3 space-y-2.5 analysis-fade-in-stagger" style={{ ["--stagger-delay" as string]: "60ms" }}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          训练阶段时间线 · PELT 变点
        </div>
        <div className="text-[10px] text-muted-foreground/70 tabular-nums">
          {segments.length} 段 · {changepointSteps.length} 个变点
        </div>
      </div>

      {/* Timeline bar. Tooltip is plain `title` so we don't need a
          floating-element library; mobile hits the segments via tap. */}
      <div className="flex h-7 w-full overflow-hidden rounded-[3px] border border-border/60 bg-muted/30">
        {segments.map((seg, i) => {
          const span = seg.endStep - seg.startStep
          const widthPct = Math.max(2, (span / totalSpan) * 100)
          return (
            <div
              key={i}
              className={cn(
                "relative h-full transition-[transform] duration-300 hover:scale-y-[1.03]",
                STAGE_COLOR[seg.stage],
                i > 0 && "border-l border-background/70",
              )}
              style={{ width: `${widthPct}%` }}
              title={[
                `阶段: ${STAGE_LABEL[seg.stage]}`,
                `步数: ${seg.startStep} → ${seg.endStep}`,
                `斜率: ${seg.slope.toExponential(2)}`,
                `均值: ${seg.meanLoss.toFixed(4)}`,
              ].join("\n")}
            >
              {widthPct > 12 && (
                <span className="absolute inset-0 grid place-items-center text-[10px] font-medium text-white/95 drop-shadow-sm">
                  {STAGE_LABEL[seg.stage]}
                </span>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[10.5px] text-muted-foreground/80">
        {segments.map((seg, i) => (
          <span key={i} className="inline-flex items-center gap-1.5">
            <span
              className={cn("inline-block size-2 rounded-full", STAGE_COLOR[seg.stage])}
              aria-hidden
            />
            <span className={cn("font-mono tabular-nums", STAGE_TEXT[seg.stage])}>
              {STAGE_LABEL[seg.stage]}
            </span>
            <span className="text-muted-foreground/70 tabular-nums">
              {seg.startStep}–{seg.endStep}
              {" · "}斜率 {seg.slope.toExponential(1)}
            </span>
          </span>
        ))}
      </div>

      <div className="text-[10.5px] text-muted-foreground/70 leading-relaxed">
        变点用 PELT 算法 (BIC 惩罚)，识别"动态显著改变"的步数点。在损失图上以青色虚线标出。
      </div>
    </div>
  )
}
