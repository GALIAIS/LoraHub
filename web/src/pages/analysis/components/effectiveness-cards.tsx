import { Sparkles } from "lucide-react"
import type { ReactNode } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import {
  STAGE_BG,
  STAGE_LABELS,
  STAGE_TONES,
  type StageKey,
} from "./effectiveness-model"
import { useAnimatedNumber } from "./use-animated-number"

export type Tone = "positive" | "neutral" | "negative"

const TONE_FILL: Record<Tone, string> = {
  positive: "bg-emerald-500/70",
  neutral: "bg-amber-500/70",
  negative: "bg-red-500/70",
}

const TONE_TEXT: Record<Tone, string> = {
  positive: "text-emerald-700 dark:text-emerald-300",
  neutral: "text-amber-700 dark:text-amber-300",
  negative: "text-red-600 dark:text-red-400",
}

const STAGE_BG_FROM_TONE: Record<Tone, string> = {
  positive: "from-emerald-500/15 to-emerald-500/0",
  neutral: "from-amber-500/15 to-amber-500/0",
  negative: "from-red-500/15 to-red-500/0",
}

interface InsightCardProps {
  icon: ReactNode
  title: string
  tone: Tone
  headline: string
  /**
   * Optional numeric value driving the headline. When supplied, the
   * card tweens this number on update with `useAnimatedNumber` and
   * uses `formatHeadline` to render the current frame. Falls back
   * to the static `headline` string when null/undefined or when
   * the format function isn't provided.
   */
  headlineNumber?: number | null
  formatHeadline?: (v: number) => string
  caption: string
  fill: number
  stagger: number
  /** Bullet-style strings shown under the "判定依据" disclosure. */
  rationale?: string[]
  /**
   * Optional flag rendered as a "低置信度" pill. Use it when the
   * verdict is computed from too few samples or noisy partial data so
   * the user knows to weigh it accordingly.
   */
  lowConfidence?: boolean
}

export function InsightCard({
  icon,
  title,
  tone,
  headline,
  headlineNumber,
  formatHeadline,
  caption,
  fill,
  stagger,
  rationale,
  lowConfidence,
}: InsightCardProps) {
  const animated = useAnimatedNumber(
    typeof headlineNumber === "number" && Number.isFinite(headlineNumber)
      ? headlineNumber
      : 0,
  )
  const animatedFill = useAnimatedNumber(fill, 280)
  const renderHeadline =
    typeof headlineNumber === "number" &&
    Number.isFinite(headlineNumber) &&
    formatHeadline
      ? formatHeadline(animated)
      : headline
  return (
    <Card
      className="analysis-fade-in-stagger overflow-hidden"
      style={{ ["--stagger-delay" as string]: `${stagger}ms` }}
    >
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[10.5px] tracking-[0.16em] text-foreground/85 font-mono inline-flex items-center gap-1.5">
          <span className={cn("opacity-80", TONE_TEXT[tone])}>{icon}</span>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3.5 space-y-2">
        <div
          className={cn(
            "text-[18px] font-semibold tracking-tight tabular-nums",
            TONE_TEXT[tone],
          )}
        >
          {renderHeadline}
        </div>
        <div className="h-1.5 rounded-full bg-muted/60 overflow-hidden">
          <div
            className={cn(
              "analysis-bar-fill h-full rounded-full transition-[background-color] duration-300",
              TONE_FILL[tone],
            )}
            style={{ width: `${(animatedFill * 100).toFixed(1)}%` }}
          />
        </div>
        <div className="text-[11px] text-muted-foreground leading-relaxed">
          {caption}
          {lowConfidence && (
            <span className="ml-1.5 inline-flex items-center rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1 py-[1px] align-middle text-[9.5px] uppercase tracking-[0.14em] text-amber-700 dark:text-amber-300">
              低置信
            </span>
          )}
        </div>
        {rationale && rationale.length > 0 && (
          <details className="group text-[10.5px] text-muted-foreground/85 mt-1.5">
            <summary className="cursor-pointer select-none inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
              <span className="inline-block transition-transform group-open:rotate-90">
                ▸
              </span>
              判定依据
            </summary>
            <ul className="mt-1 space-y-0.5 pl-3 leading-relaxed">
              {rationale.map((r, i) => (
                <li key={i} className="font-mono">
                  · {r}
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  )
}

interface StageCardProps {
  stage: StageKey
  /**
   * Optional tone override that lets the parent inject context-aware
   * colour (e.g. plateau in the late phase = positive, plateau in
   * the early phase = negative). Falls back to the per-stage default.
   */
  stageTone?: Tone
  reason: string
  stagger: number
  rationale?: string[]
}

export function StageCard({
  stage,
  stageTone,
  reason,
  stagger,
  rationale,
}: StageCardProps) {
  const textCls = stageTone ? TONE_TEXT[stageTone] : STAGE_TONES[stage]
  const bgCls = stageTone ? STAGE_BG_FROM_TONE[stageTone] : STAGE_BG[stage]
  return (
    <Card
      className="analysis-fade-in-stagger overflow-hidden relative"
      style={{ ["--stagger-delay" as string]: `${stagger}ms` }}
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-0 bg-gradient-to-br opacity-70",
          bgCls,
        )}
        aria-hidden
      />
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2 relative">
        <CardTitle className="text-[10.5px] tracking-[0.16em] text-foreground/85 font-mono inline-flex items-center gap-1.5">
          <span className={cn("opacity-80", textCls)}>
            <Sparkles className="size-3.5" />
          </span>
          训练阶段
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3.5 space-y-2 relative">
        <div
          className={cn("text-[18px] font-semibold tracking-tight", textCls)}
        >
          {STAGE_LABELS[stage]}
        </div>
        <StageDots stage={stage} />
        <div className="text-[11px] text-muted-foreground leading-relaxed min-h-[1.2em]">
          {reason || "等待更多训练数据"}
        </div>
        {rationale && rationale.length > 0 && (
          <details className="group text-[10.5px] text-muted-foreground/85 mt-1">
            <summary className="cursor-pointer select-none inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
              <span className="inline-block transition-transform group-open:rotate-90">
                ▸
              </span>
              判定依据
            </summary>
            <ul className="mt-1 space-y-0.5 pl-3 leading-relaxed">
              {rationale.map((r, i) => (
                <li key={i} className="font-mono">
                  · {r}
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  )
}

const STAGE_ORDER: StageKey[] = ["warmup", "converging", "plateau", "diverging"]

function StageDots({ stage }: { stage: StageKey }) {
  const idx = STAGE_ORDER.indexOf(stage)
  return (
    <div
      className="flex items-center gap-1.5"
      aria-label={`阶段: ${STAGE_LABELS[stage]}`}
    >
      {STAGE_ORDER.map((s, i) => {
        const isActive = s === stage
        const isPassed = idx >= 0 && i < idx
        return (
          <span
            key={s}
            className={cn(
              "inline-block h-1.5 flex-1 rounded-full transition-all duration-500",
              isActive
                ? "bg-foreground/85 scale-y-[1.4]"
                : isPassed
                  ? "bg-foreground/40"
                  : "bg-muted/60",
            )}
            aria-hidden
          />
        )
      })}
    </div>
  )
}
