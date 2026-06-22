/**
 * "训练分析 → W&B" tab. Read-only view of the wandb run associated
 * with the current job, served by the backend wandb-public-API proxy
 * at `/api/wandb/*`.
 *
 * Why this tab exists at all
 * --------------------------
 * The wandb run dashboard cannot be embedded in an iframe (wandb sets
 * `X-Frame-Options: DENY`), so we cannot drop their UI into LoraHub.
 * Instead the backend pulls the metrics through `wandb.Api()` and
 * this component renders them with the same `MultiLineChart` widget
 * the rest of the workbench uses. Identity (project, entity, tags,
 * url) is rendered alongside so users have parity with the wandb UI
 * for the most-asked-about fields.
 */
import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  AlertCircle,
  ExternalLink,
  Loader2,
  Settings as SettingsIcon,
} from "lucide-react"
import { Link } from "react-router-dom"
import { wandbApi } from "@/lib/api"
import type { WandbHistoryResponse } from "@/lib/api"
import {
  MultiLineChart,
  type MultiLineSeries,
} from "../components/multi-line-chart"

const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
]

interface WandbTabProps {
  jobId: string
  /** When false, the user hasn't enabled `monitoring.enable_wandb` for this run. */
  enabled: boolean
  /** When falsy, the run hasn't reached `wandb.init()` yet (no URL stamped). */
  runUrl?: string | null
}

export function WandbTab({ jobId, enabled, runUrl }: WandbTabProps) {
  const status = useQuery({
    queryKey: ["wandb", "status"],
    queryFn: wandbApi.status,
    staleTime: 30_000,
    retry: false,
  })

  const ready = enabled && Boolean(runUrl)

  const summary = useQuery({
    queryKey: ["wandb", "summary", jobId],
    queryFn: () => wandbApi.runSummary(jobId),
    enabled: ready && status.data?.installed === true && status.data.api_key_configured === true,
    staleTime: 30_000,
    retry: false,
  })

  const history = useQuery({
    queryKey: ["wandb", "history", jobId],
    queryFn: () => wandbApi.runHistory(jobId, { samples: 1000 }),
    enabled: ready && status.data?.installed === true && status.data.api_key_configured === true,
    staleTime: 30_000,
    retry: false,
  })

  if (!enabled) {
    return (
      <EmptyHint
        title="未启用 W&B"
        body="在配置编辑器的「实验跟踪」中勾选 启用 W&B 后,新启动的训练会推送到 wandb.ai。"
      />
    )
  }

  if (!runUrl) {
    return (
      <EmptyHint
        title="W&B run 尚未启动"
        body="训练子进程还未到达 wandb.init();等待 run url 写入后此处会自动展示数据。"
      />
    )
  }

  if (status.isLoading) {
    return <LoadingHint />
  }

  if (status.data && !status.data.installed) {
    return (
      <EmptyHint
        title="API 端未安装 wandb"
        body="服务端环境缺少 wandb 包。pip install wandb 后重启 LoraHub API 即可使用。"
      />
    )
  }

  if (status.data && !status.data.api_key_configured) {
    return (
      <EmptyHint
        title="未配置 W&B API Key"
        body={
          <>
            前往 <SettingsLink /> 填写 API Key 后,即可在此查看 wandb 数据。
          </>
        }
      />
    )
  }

  if (summary.isError || history.isError) {
    const err = (summary.error ?? history.error) as unknown
    return (
      <EmptyHint
        title="拉取 wandb 数据失败"
        body={String((err as Error)?.message ?? err)}
        icon={<AlertCircle className="size-5 text-destructive" />}
      />
    )
  }

  if (summary.isLoading || history.isLoading) {
    return <LoadingHint />
  }

  return (
    <div className="grid gap-4 p-3.5">
      <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1.5 text-[12px]">
        <div>
          <span className="text-muted-foreground">project </span>
          <code className="font-mono text-foreground">
            {summary.data?.entity}/{summary.data?.project}
          </code>
        </div>
        {summary.data?.name && (
          <div>
            <span className="text-muted-foreground">run </span>
            <code className="font-mono text-foreground">{summary.data.name}</code>
          </div>
        )}
        {summary.data?.state && (
          <div>
            <span className="text-muted-foreground">state </span>
            <span className="font-mono">{summary.data.state}</span>
          </div>
        )}
        {summary.data?.tags && summary.data.tags.length > 0 && (
          <div className="flex flex-wrap items-center gap-1">
            {summary.data.tags.map((t) => (
              <span
                key={t}
                className="px-1.5 py-0.5 rounded-[3px] bg-muted/50 text-[10.5px] tabular-nums text-foreground/80"
              >
                {t}
              </span>
            ))}
          </div>
        )}
        <a
          href={summary.data?.url ?? runUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto inline-flex items-center gap-1 text-[11.5px] text-muted-foreground hover:text-foreground"
        >
          <ExternalLink className="size-3.5" /> 在 wandb.ai 打开
        </a>
      </header>

      <ChartsGrid history={history.data ?? null} />

      {summary.data?.summary && Object.keys(summary.data.summary).length > 0 && (
        <SummaryGrid summary={summary.data.summary} />
      )}
    </div>
  )
}

function ChartsGrid({ history }: { history: WandbHistoryResponse | null }) {
  const seriesGroups = useMemo(() => buildSeriesGroups(history), [history])
  if (!history || history.rows.length === 0) {
    return (
      <EmptyHint
        title="W&B 暂无 history"
        body="run 已注册但还未上报任何 step 数据。等首次 log 后再回来。"
      />
    )
  }
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {seriesGroups.map((group) => (
        <div key={group.title} className="rounded-[6px] border border-border/40 bg-background/40 p-2">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground px-1 pt-0.5 pb-1.5">
            {group.title}
          </div>
          <MultiLineChart
            series={group.series}
            xLabel="step"
            persistKey={`wandb-${group.title}`}
          />
        </div>
      ))}
    </div>
  )
}

function SummaryGrid({ summary }: { summary: Record<string, unknown> }) {
  const entries = Object.entries(summary)
    .filter(([, v]) => typeof v === "number" || typeof v === "string")
    .sort(([a], [b]) => a.localeCompare(b))
  if (entries.length === 0) return null
  return (
    <div className="rounded-[6px] border border-border/40 bg-background/40 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground pb-2">
        Summary
      </div>
      <dl className="grid gap-x-6 gap-y-1 grid-cols-[max-content_1fr] text-[11.5px]">
        {entries.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="font-mono text-muted-foreground">{k}</dt>
            <dd className="font-mono tabular-nums text-foreground">
              {typeof v === "number" ? formatNumber(v) : String(v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function buildSeriesGroups(
  history: WandbHistoryResponse | null,
): Array<{ title: string; series: MultiLineSeries[] }> {
  if (!history || history.rows.length === 0) return []
  // Wandb's pandas history always carries `_step` as the x axis.
  const xKey = history.keys.includes("_step") ? "_step" : history.keys[0]
  const numericKeys = history.keys.filter(
    (k) =>
      k !== xKey &&
      !k.startsWith("_") &&
      history.rows.some((r) => typeof r[k] === "number"),
  )
  // Group keys into a few semantic buckets so users don't drown in
  // 30 lines on one chart. Rest fall into "其他".
  const groups: Record<string, string[]> = {
    Loss: numericKeys.filter((k) => /loss|val_loss/i.test(k)),
    "Learning Rate": numericKeys.filter((k) => /\blr\b|learning_rate/i.test(k)),
    Gradient: numericKeys.filter((k) => /grad|norm/i.test(k)),
    System: numericKeys.filter((k) => /^system\.|gpu|cpu|memory/i.test(k)),
  }
  const claimed = new Set<string>()
  for (const list of Object.values(groups)) for (const k of list) claimed.add(k)
  const rest = numericKeys.filter((k) => !claimed.has(k))
  if (rest.length > 0) groups["其他"] = rest

  const out: Array<{ title: string; series: MultiLineSeries[] }> = []
  for (const [title, keys] of Object.entries(groups)) {
    if (keys.length === 0) continue
    const series: MultiLineSeries[] = keys.map((k, i) => ({
      id: k,
      label: k,
      color: SERIES_COLORS[i % SERIES_COLORS.length]!,
      points: history.rows
        .map((r) => {
          const xv = r[xKey!]
          const yv = r[k]
          if (typeof xv !== "number" || typeof yv !== "number") return null
          return { x: xv, y: yv }
        })
        .filter((p): p is { x: number; y: number } => p !== null),
    }))
    out.push({ title, series })
  }
  return out
}

function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001)) {
    return v.toExponential(3)
  }
  return v.toFixed(4).replace(/\.?0+$/, "")
}

function SettingsLink() {
  return (
    <Link
      to="/settings/network"
      className="inline-flex items-center gap-1 text-foreground hover:underline"
    >
      <SettingsIcon className="size-3.5" /> 设置 → 网络
    </Link>
  )
}

function EmptyHint({
  title,
  body,
  icon,
}: {
  title: string
  body: React.ReactNode
  icon?: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 px-6 text-center">
      {icon ?? <AlertCircle className="size-5 text-muted-foreground" />}
      <div className="text-[13px] font-medium">{title}</div>
      <div className="text-[12px] text-muted-foreground max-w-md leading-relaxed">
        {body}
      </div>
    </div>
  )
}

function LoadingHint() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12">
      <Loader2 className="size-5 animate-spin text-muted-foreground" />
      <div className="text-[12px] text-muted-foreground">正在拉取 wandb 数据...</div>
    </div>
  )
}
