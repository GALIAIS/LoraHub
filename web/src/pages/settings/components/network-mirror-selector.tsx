import { useEffect, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { Loader2, Sparkles, Zap } from "lucide-react"
import { api, type MirrorPreset, type ProbeResult } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—"
  if (ms < 100) return `${ms.toFixed(0)} ms`
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

function latencyTone(ms: number | null | undefined, ok: boolean): string {
  if (!ok || ms === null || ms === undefined) return "text-destructive"
  if (ms < 200) return "text-emerald-600 dark:text-emerald-400"
  if (ms < 500) return "text-primary"
  if (ms < 1500) return "text-amber-600 dark:text-amber-400"
  return "text-destructive"
}

interface MirrorSelectorProps {
  category: "github_proxy" | "huggingface" | "pypi" | "pytorch"
  presets: MirrorPreset[]
  current: string
  onChoose: (value: string) => void
  /** Called when the user clicks "测速并自动选用最快" with a fresh result. */
  onAutoPick?: (result: ProbeResult) => void
}

export function MirrorSelector({
  category,
  presets,
  current,
  onChoose,
  onAutoPick,
}: MirrorSelectorProps) {
  const [results, setResults] = useState<ProbeResult[] | null>(null)

  const probe = useMutation({
    mutationFn: () => api.probeMirrors({ category }),
    onSuccess: (rows) => setResults(rows),
  })

  // Auto-pick the fastest reachable mirror once the probe lands.
  useEffect(() => {
    if (!probe.isSuccess || !probe.data) return
    const fastest = probe.data.find((r) => r.ok)
    if (fastest) {
      onChoose(fastest.value)
      onAutoPick?.(fastest)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [probe.isSuccess])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          可选镜像
        </span>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={probe.isPending}
          onClick={() => {
            setResults(null)
            probe.mutate()
          }}
        >
          {probe.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Zap className="size-3" />
          )}
          {probe.isPending ? "测速中…" : "测速并自动选用最快"}
        </Button>
      </div>

      <div className="rounded-[4px] border border-border/60 divide-y divide-border/40 overflow-hidden">
        {presets.map((preset) => {
          const result = results?.find((row) => row.value === preset.value)
          const fastest =
            results !== null && results.length > 0 && results.find((row) => row.ok)?.value
          const isCurrent = current === preset.value
          const isFastest =
            result && result.value === fastest && result.ok
          return (
            <button
              key={preset.label + preset.value}
              type="button"
              onClick={() => onChoose(preset.value)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 text-xs text-left transition-colors",
                isCurrent
                  ? "bg-primary/10 text-foreground"
                  : "hover:bg-muted/50 text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="flex-1 min-w-0">
                <span className="block font-medium truncate">
                  {preset.label}
                </span>
                {preset.value && (
                  <span className="block text-[10px] font-mono text-muted-foreground/70 truncate">
                    {preset.value}
                  </span>
                )}
              </span>
              {result && (
                <span
                  className={cn(
                    "text-[11px] font-mono tabular-nums shrink-0",
                    latencyTone(result.latency_ms, result.ok),
                  )}
                  title={result.error ?? undefined}
                >
                  {result.ok ? formatLatency(result.latency_ms) : "不可达"}
                </span>
              )}
              {isFastest && (
                <span className="rounded-[2px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.1em] shrink-0 inline-flex items-center gap-1">
                  <Sparkles className="size-2.5" />
                  最快
                </span>
              )}
              {isCurrent && (
                <span className="rounded-[2px] bg-primary/10 text-primary px-1.5 py-0.5 text-[10px] uppercase tracking-[0.1em] shrink-0">
                  已选
                </span>
              )}
            </button>
          )
        })}
      </div>

      {probe.isError && (
        <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-1.5 text-xs font-mono text-destructive">
          {(probe.error as Error).message}
        </div>
      )}
    </div>
  )
}
