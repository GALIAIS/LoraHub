import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export const ACTIVE_STATES = new Set(["queued", "running", "canceling"])

// State color tokens for the sweep distribution mini-bar — kept literal here
// so we don't have to spin up a tailwind plugin entry just for one widget.
export const STATE_COLORS: Record<string, string> = {
  succeeded: "bg-emerald-500/85",
  running: "bg-sky-500/85",
  queued: "bg-muted-foreground/40",
  failed: "bg-rose-500/85",
  canceled: "bg-amber-500/70",
  canceling: "bg-amber-500/70",
  interrupted: "bg-rose-500/60",
}

const MODE_BADGE: Record<string, { label: string; toneClass: string }> = {
  grid: {
    label: "grid",
    toneClass:
      "border-zinc-500/40 bg-zinc-500/10 text-zinc-700 dark:text-zinc-300",
  },
  random: {
    label: "random",
    toneClass:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  },
  tpe: {
    label: "TPE",
    toneClass:
      "border-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
  },
}

export function ModeBadge({ mode }: { mode: string | undefined | null }) {
  const meta = MODE_BADGE[mode ?? "grid"] ?? MODE_BADGE.grid
  return (
    <Badge
      variant="outline"
      className={cn(
        "rounded-[2px] uppercase text-[10px] tracking-[0.1em]",
        meta.toneClass,
      )}
    >
      {meta.label}
    </Badge>
  )
}

export function fmtRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "—"
  const ts = new Date(iso).getTime()
  if (!Number.isFinite(ts)) return "—"
  const delta = (Date.now() - ts) / 1000
  if (delta < 60) return "刚刚"
  if (delta < 3600) return `${Math.floor(delta / 60)} 分钟前`
  if (delta < 86400) return `${Math.floor(delta / 3600)} 小时前`
  if (delta < 86400 * 30) return `${Math.floor(delta / 86400)} 天前`
  return new Date(ts).toLocaleDateString()
}

export function formatAxisValue(value: unknown): string {
  if (value === null) return "null"
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}
