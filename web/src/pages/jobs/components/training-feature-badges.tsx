/**
 * Training-features badge strip — visual confirmation that the
 * trainer-side toggles the user enabled (EMA, NaN guard, Min-SNR-γ,
 * sample grid, …) actually made it into the run's config snapshot.
 *
 * Inspecting features post-launch usually requires opening the YAML.
 * Surfacing them as small chips on the overview tab catches the
 * "I thought I enabled X" mistake without forcing the user to dig.
 *
 * The component reads ``config_snapshot.backend.animaLora`` (the
 * camelCase keys the API exposes) and quietly renders nothing when
 * the snapshot is unavailable / non-anima.
 */
import {
  CheckCircle2,
  ImageDown,
  Minus,
  Shield,
  Sparkles,
  TrendingUp,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface FeatureBadgeProps {
  configSnapshot?: Record<string, unknown>
}

interface Feature {
  key: string
  label: string
  detail?: string
  icon: React.ComponentType<{ className?: string }>
  active: boolean
}

export function TrainingFeatureBadges({ configSnapshot }: FeatureBadgeProps) {
  const anima = readAnimaLora(configSnapshot)
  if (!anima) return null

  const features: Feature[] = [
    {
      key: "ema",
      label: "EMA",
      detail: extractEmaDetail(anima),
      icon: TrendingUp,
      active: Boolean(anima.ema),
    },
    {
      key: "nan_guard",
      label: "NaN Guard",
      detail: anima.nan_guard_recover ? "+ recover" : undefined,
      icon: Shield,
      active: Boolean(anima.nan_guard),
    },
    {
      key: "min_snr",
      label: "Min-SNR γ",
      detail: extractMinSnrDetail(anima),
      icon: Sparkles,
      active:
        anima.weighting_scheme === "min_snr_rf" &&
        typeof anima.min_snr_gamma === "number" &&
        anima.min_snr_gamma > 0,
    },
    {
      key: "sample_grid",
      label: "Sample Grid",
      icon: ImageDown,
      active: Boolean(anima.sample_grid),
    },
  ]

  // No badges to surface? Hide the strip entirely so the overview
  // doesn't waste vertical space.
  const hasAny = features.some((f) => f.active)
  if (!hasAny) return null

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/80 mr-1">
        训练特性
      </span>
      {features.map((f) => {
        const Icon = f.icon
        const tone = f.active
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border/60 bg-muted/30 text-muted-foreground"
        return (
          <Badge
            key={f.key}
            variant="outline"
            className={cn(
              "rounded-[2px] gap-1.5 text-[11px] py-0.5 px-1.5",
              tone,
            )}
            title={f.active ? `已启用${f.detail ? ` (${f.detail})` : ""}` : "未启用"}
          >
            {f.active ? (
              <CheckCircle2 className="size-3" />
            ) : (
              <Minus className="size-3 opacity-60" />
            )}
            <Icon className="size-3" />
            <span className="font-mono">
              {f.label}
              {f.detail && f.active && (
                <span className="text-muted-foreground/80 ml-1">
                  {f.detail}
                </span>
              )}
            </span>
          </Badge>
        )
      })}
    </div>
  )
}

function readAnimaLora(snapshot: Record<string, unknown> | undefined): Record<string, unknown> | null {
  if (!snapshot) return null
  const backend = snapshot.backend as Record<string, unknown> | undefined
  if (!backend) return null
  const anima = backend.animaLora as Record<string, unknown> | undefined
  if (!anima) return null
  return anima
}

function extractEmaDetail(anima: Record<string, unknown>): string | undefined {
  if (!anima.ema) return undefined
  const decay = anima.ema_decay
  if (typeof decay === "number") {
    return `decay ${decay}`
  }
  return undefined
}

function extractMinSnrDetail(anima: Record<string, unknown>): string | undefined {
  const γ = anima.min_snr_gamma
  if (typeof γ === "number" && γ > 0) {
    return `γ=${γ}`
  }
  return undefined
}
