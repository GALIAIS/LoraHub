import { AlertTriangle } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import type { JobMetricsResponse } from "@/lib/api"
import { cn } from "@/lib/utils"
import type { AnalysisBackendInfo } from "./analysis-backend"

export function BackendContextStrip({
  backend,
  metrics,
}: {
  backend: AnalysisBackendInfo
  metrics: JobMetricsResponse | null
}) {
  const hasEpoch = (metrics?.loss ?? []).some(
    (point) => typeof point.epoch === "number",
  )
  const hasValidation = (metrics?.val_loss?.length ?? 0) > 0
  const hasThroughput = (metrics?.loss ?? []).some(
    (point) => typeof point.iter_time_s === "number",
  )
  const cachePhases = summarizeCache(metrics?.cache_progress ?? [])
  const sampleFailures = (metrics?.diagnostics ?? []).filter((item) =>
    /sample|preview/i.test(item.category ?? ""),
  ).length
  const dangerousLr =
    backend.type === "anima_lora" &&
    backend.configuredLr != null &&
    backend.configuredLr > 1e-2

  return (
    <div className="flex min-h-9 flex-wrap items-center gap-x-3 gap-y-1.5 border-y border-border/55 bg-muted/20 px-3.5 py-1.5 text-[10.5px] text-muted-foreground">
      <Badge
        variant="outline"
        className="h-5 rounded-[3px] px-1.5 font-mono normal-case tracking-normal"
      >
        {backend.label}
      </Badge>
      {backend.type === "anima_lora" && <Signal label="Loss" value="avr_loss" />}
      {backend.type === "ai_toolkit" && <Signal label="Loss" value="step loss" />}
      <Signal
        label="验证"
        value={
          hasValidation
            ? "已上报"
            : backend.supportsValidation === false
              ? "后端未提供"
              : backend.validationConfigured === false
                ? "未配置"
                : "等待数据"
        }
      />
      <Signal
        label="Epoch"
        value={hasEpoch ? "已上报" : "后端未上报，按 step 分析"}
      />
      <Signal
        label="吞吐"
        value={hasThroughput ? "实时" : "等待后端上报"}
      />
      {backend.gradAccum != null && (
        <Signal label="梯度累积" value={`×${backend.gradAccum}`} />
      )}
      {backend.minSnrGamma != null && (
        <Signal label="Min-SNR γ" value={formatNumber(backend.minSnrGamma)} />
      )}
      {backend.type === "anima_lora" && backend.flowShift != null && (
        <Signal label="flow shift" value={formatNumber(backend.flowShift)} />
      )}
      {backend.type === "anima_lora" &&
        (backend.sampler || backend.sampleSteps || backend.sampleCfg) && (
          <Signal
            label="采样"
            value={`${backend.sampler ?? "默认采样器"}${backend.sampleSteps ? ` · ${backend.sampleSteps} 步` : ""}${backend.sampleCfg ? ` · CFG ${backend.sampleCfg}` : ""}`}
          />
        )}
      {cachePhases.map((phase) => (
        <Signal
          key={phase.phase ?? "cache"}
          label="预处理"
          value={phase.value}
        />
      ))}
      {sampleFailures > 0 && (
        <span className="inline-flex items-center gap-1 text-destructive">
          <AlertTriangle className="size-3" />
          采样诊断 {sampleFailures}
        </span>
      )}
      {dangerousLr && (
        <span
          className="inline-flex items-center gap-1 text-destructive"
          title="Anima LoRA 的学习率通常远低于 1e-2"
        >
          <AlertTriangle className="size-3" />
          LR {backend.configuredLr?.toExponential(2)} 量级异常
        </span>
      )}
    </div>
  )
}

function Signal({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      <span className="text-muted-foreground/65">{label}</span>
      <span
        className={cn(
          "text-foreground/80",
          value.includes("未") && "text-muted-foreground",
        )}
      >
        {value}
      </span>
    </span>
  )
}

function cacheLabel(phase: string | null): string {
  if (phase === "latents") return "潜空间缓存"
  if (phase === "text_encoder") return "文本编码缓存"
  return phase || "缓存"
}

function formatProgress(
  progress: JobMetricsResponse["cache_progress"][number],
): string {
  if (typeof progress.done === "number" && typeof progress.total === "number") {
    return `${progress.done}/${progress.total}`
  }
  if (typeof progress.percent === "number") return `${progress.percent}%`
  return ""
}

function summarizeCache(
  points: JobMetricsResponse["cache_progress"],
): Array<{ phase: string | null; value: string }> {
  const phases = new Map<
    string | null,
    {
      firstTs: number
      lastTs: number
      latest: JobMetricsResponse["cache_progress"][number]
    }
  >()
  for (const point of points) {
    const current = phases.get(point.phase)
    phases.set(point.phase, {
      firstTs: current?.firstTs ?? point.ts,
      lastTs: point.ts,
      latest: point,
    })
  }
  return Array.from(phases.entries()).map(([phase, summary]) => {
    const elapsed = Math.max(0, summary.lastTs - summary.firstTs)
    const duration = elapsed >= 1 ? ` · ${formatElapsed(elapsed)}` : ""
    return {
      phase,
      value: `${cacheLabel(phase)} ${formatProgress(summary.latest)}${duration}`,
    }
  })
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} 秒`
  return `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}
