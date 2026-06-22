import { Activity, PanelLeftClose } from "lucide-react"

import type { SweepSummary } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import {
  fmtRelativeTime,
  ModeBadge,
  STATE_COLORS,
} from "./sweep-shared"

export function SweepSidebar({
  list,
  loading,
  selectedId,
  onSelect,
  onClose,
}: {
  list: SweepSummary[]
  loading: boolean
  selectedId: string | null
  onSelect: (id: string) => void
  onClose: () => void
}) {
  return (
    <>
      <div className="flex items-center justify-between px-4 pt-3">
        <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          参数搜索
        </span>
        <Button
          size="sm"
          variant="ghost"
          onClick={onClose}
          title="收起侧栏"
        >
          <PanelLeftClose className="size-4" />
        </Button>
      </div>
      <div className="px-4 pt-1 pb-3 text-[11px] text-muted-foreground">
        {loading ? "加载中…" : `${list.length} 个 sweep`}
      </div>
      <ScrollArea className="flex-1 min-h-0">
        <ul className="divide-y divide-border/40">
          {!loading && list.length === 0 && (
            <li className="px-5 py-10 text-sm text-muted-foreground text-center">
              还没有 sweep。运行
              <code className="text-foreground"> lorahub sweep config.yaml --axis ...</code>
              {" "}或 POST{" "}
              <code className="text-foreground">/api/sweeps</code> 触发一次。
            </li>
          )}
          {list.map((s) => (
            <SweepListRow
              key={s.sweep_id}
              sweep={s}
              active={s.sweep_id === selectedId}
              onSelect={() => onSelect(s.sweep_id)}
            />
          ))}
        </ul>
      </ScrollArea>
    </>
  )
}

function SweepListRow({
  sweep,
  active,
  onSelect,
}: {
  sweep: SweepSummary
  active: boolean
  onSelect: () => void
}) {
  const isActive = sweep.queued + sweep.running + sweep.canceling > 0
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          "w-full text-left px-4 py-3 transition-colors flex flex-col gap-1.5",
          active
            ? "bg-sidebar-accent/70 text-foreground"
            : "hover:bg-muted/40 text-muted-foreground",
        )}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-medium text-foreground truncate">
            {sweep.name_prefix || sweep.sweep_id.slice(-8)}
          </span>
          <ModeBadge mode={sweep.mode} />
        </div>
        <DistributionBar sweep={sweep} />
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>
            {sweep.total} 个变体
            {isActive && (
              <span className="ml-1.5 inline-flex items-center gap-1 text-foreground/80">
                <Activity className="size-3" />
                进行中
              </span>
            )}
          </span>
          <span className="tabular-nums">
            {fmtRelativeTime(sweep.latest_modified_at)}
          </span>
        </div>
      </button>
    </li>
  )
}

function DistributionBar({ sweep }: { sweep: SweepSummary }) {
  const segments: Array<{ key: string; count: number; label: string }> = [
    { key: "succeeded", count: sweep.succeeded, label: "已完成" },
    { key: "running", count: sweep.running, label: "运行中" },
    { key: "queued", count: sweep.queued, label: "排队中" },
    { key: "failed", count: sweep.failed, label: "失败" },
    {
      key: "canceled",
      count: sweep.canceled + sweep.canceling,
      label: "已取消",
    },
    { key: "interrupted", count: sweep.interrupted, label: "中断" },
  ]
  const total = sweep.total || 1
  return (
    <div
      className="shiro-progress-track h-1.5 w-full flex"
      title={segments
        .filter((s) => s.count > 0)
        .map((s) => `${s.label} ${s.count}`)
        .join(" · ")}
    >
      {segments.map((seg) =>
        seg.count > 0 ? (
          <div
            key={seg.key}
            className={cn("shiro-progress-fill", STATE_COLORS[seg.key])}
            style={{ width: `${(seg.count / total) * 100}%` }}
          />
        ) : null,
      )}
    </div>
  )
}
