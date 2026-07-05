import { memo } from "react"
import { AlertTriangle, Thermometer, Zap } from "lucide-react"
import type { SystemGpu } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { Metric, UsageBar, fmtBytes, toneForPercent } from "./_shared"

export const GpuSection = memo(function GpuSection({
  gpus,
  hasNvidiaSmi,
  system,
}: {
  gpus: SystemGpu[]
  hasNvidiaSmi: boolean
  system: string
}) {
  // 已经探测到设备时直接渲染，让多源 GPU 都有展示。
  if (gpus.length > 0) {
    return (
      <div className="flex h-full flex-col gap-3">
        {gpus.map((gpu) => (
          <GpuCard key={gpu.index} gpu={gpu} />
        ))}
      </div>
    )
  }

  const isMac = system === "Darwin"
  const description = isMac
    ? "Apple Silicon / Apple GPU 不支持详细计量，已切换到仅展示型号。本机未检测到独立 GPU。"
    : "未检测到 nvidia-smi。AMD / Apple 设备或 CPU-only 主机可忽略此项。"

  // 没有任何 GPU 数据 - 给个友好提示。
  if (!hasNvidiaSmi) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="size-4 text-muted-foreground" />
            GPU
          </CardTitle>
          <CardDescription className="text-xs">{description}</CardDescription>
        </CardHeader>
      </Card>
    )
  }
  return (
    <Card className="h-full rounded-[6px] border-amber-500/30 bg-amber-500/5 shadow-[var(--panel-shadow)]">
      <CardContent className="px-4 py-3 flex items-center gap-2 text-xs text-amber-700 dark:text-amber-400">
        <AlertTriangle className="size-4" />
        nvidia-smi 已安装但未返回任何设备信息。
      </CardContent>
    </Card>
  )
})

const GpuCard = memo(function GpuCard({ gpu }: { gpu: SystemGpu }) {
  const memPercent =
    typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
      ? Math.max(0, Math.min(100, (gpu.memory_used_bytes / gpu.memory_total_bytes) * 100))
      : null
  const utilTone = toneForPercent(gpu.utilization_percent ?? 0)
  const memTone = toneForPercent(memPercent)
  const vendor = vendorBadge(gpu)
  const isAppleSilicon = gpu.vendor === "apple"
  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="size-4 text-muted-foreground" />
              GPU #{gpu.index} · {gpu.name}
            </CardTitle>
            <CardDescription className="text-xs">
              {isAppleSilicon
                ? "Apple GPU 暂不开放计量接口，仅显示型号"
                : `驱动 ${gpu.driver ?? "—"}`}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={cn("rounded-[2px] uppercase text-[10px] tracking-[0.1em]", vendor.className)}>
              {vendor.label}
            </Badge>
            {typeof gpu.temperature_c === "number" && (
              <Badge variant="outline" className="rounded-[2px] gap-1">
                <Thermometer className="size-3" /> {gpu.temperature_c.toFixed(0)}°C
              </Badge>
            )}
            {typeof gpu.fan_percent === "number" && (
              <Badge variant="outline" className="rounded-[2px]">
                风扇 {gpu.fan_percent.toFixed(0)}%
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <UsageBar
          label="计算利用率"
          percent={gpu.utilization_percent}
          valueText={
            typeof gpu.utilization_percent === "number"
              ? `${gpu.utilization_percent.toFixed(0)}%`
              : "—"
          }
        />
        <UsageBar
          label="显存"
          percent={memPercent}
          valueText={
            typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
              ? `${fmtBytes(gpu.memory_used_bytes)} / ${fmtBytes(gpu.memory_total_bytes)}`
              : gpu.memory_total_bytes
                ? `— / ${fmtBytes(gpu.memory_total_bytes)}`
                : "—"
          }
        />
        <Separator />
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
          <Metric
            label="功率"
            value={typeof gpu.power_w === "number" ? `${gpu.power_w.toFixed(0)} W` : "—"}
          />
          <Metric
            label="功率上限"
            value={
              typeof gpu.power_limit_w === "number" ? `${gpu.power_limit_w.toFixed(0)} W` : "—"
            }
          />
          <Metric
            label="可用显存"
            value={typeof gpu.memory_free_bytes === "number" ? fmtBytes(gpu.memory_free_bytes) : "—"}
          />
          <Metric
            label="温度"
            value={
              typeof gpu.temperature_c === "number" ? `${gpu.temperature_c.toFixed(0)} °C` : "—"
            }
          />
        </dl>
        {(gpu.pcie_gen_current != null ||
          gpu.pcie_width_current != null ||
          gpu.pcie_gen_max != null ||
          gpu.pcie_width_max != null ||
          gpu.sm_clock_mhz != null ||
          gpu.sm_clock_max_mhz != null ||
          gpu.mem_clock_mhz != null ||
          gpu.mem_clock_max_mhz != null) && (
          <>
            <Separator />
            <dl className="grid grid-cols-1 sm:grid-cols-3 gap-x-4 gap-y-2 text-xs">
              <Metric
                label="PCIe 链路"
                value={formatPcieLink(
                  gpu.pcie_gen_current ?? null,
                  gpu.pcie_width_current ?? null,
                  gpu.pcie_gen_max ?? null,
                  gpu.pcie_width_max ?? null,
                )}
              />
              <Metric
                label="SM 时钟"
                value={formatClockPair(
                  gpu.sm_clock_mhz ?? null,
                  gpu.sm_clock_max_mhz ?? null,
                )}
              />
              <Metric
                label="Mem 时钟"
                value={formatClockPair(
                  gpu.mem_clock_mhz ?? null,
                  gpu.mem_clock_max_mhz ?? null,
                )}
              />
            </dl>
          </>
        )}
        <span className="sr-only">
          {utilTone.text}
          {memTone.text}
        </span>
      </CardContent>
    </Card>
  )
})

function vendorBadge(gpu: SystemGpu): { label: string; className: string } {
  switch (gpu.vendor) {
    case "nvidia":
      return {
        label: "NVIDIA",
        className: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/40",
      }
    case "amd":
      return {
        label: "AMD",
        className: "bg-red-500/15 text-red-700 dark:text-red-300 border border-red-500/40",
      }
    case "intel":
      return {
        label: "Intel",
        className: "bg-blue-500/15 text-blue-700 dark:text-blue-300 border border-blue-500/40",
      }
    case "apple":
      return {
        label: "Apple",
        className: "bg-slate-500/15 text-slate-700 dark:text-slate-200 border border-slate-500/40",
      }
    default:
      return {
        label: gpu.vendor ? gpu.vendor.toUpperCase() : "GPU",
        className: "bg-muted text-foreground border border-border/70",
      }
  }
}

function formatPcieLink(
  gen: number | null,
  width: number | null,
  genMax: number | null,
  widthMax: number | null,
): string {
  const cur = gen != null && width != null ? `Gen ${gen} ×${width}` : "—"
  const max =
    genMax != null && widthMax != null ? `max Gen ${genMax} ×${widthMax}` : null
  return max ? `${cur} / ${max}` : cur
}

function formatClockPair(current: number | null, max: number | null): string {
  const cur =
    typeof current === "number" && current > 0 ? `${current.toFixed(0)} MHz` : "—"
  const mx = typeof max === "number" && max > 0 ? ` / max ${max.toFixed(0)} MHz` : ""
  return `${cur}${mx}`
}
