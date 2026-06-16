import {
  Archive,
  BarChart3,
  FolderOpen,
  GitBranch,
  Pause,
  Pencil,
  Play,
  RefreshCw,
  Skull,
  Square,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { PathDisplay } from "@/components/path-display"
import type { JobDetail as JobDetailData } from "@/lib/api"
import { StateBadge } from "../../dashboard"

type BusyAction =
  | null
  | "rerun"
  | "reveal"
  | "archive"
  | "kill"
  | "pause"
  | "resume"

type JobDetailHeaderProps = {
  actionError: string | null
  busy: BusyAction
  compareIds: string[]
  data: JobDetailData | undefined
  isLive: boolean
  isPaused: boolean
  isResumable: boolean
  isTerminal: boolean
  jobId: string
  showCompareJumpButton: boolean
  streamStatus: "idle" | "connecting" | "open" | "closed"
  onArchiveOpen: () => void
  onCancel: () => void
  onCloneOpen: () => void
  onKillOpen: () => void
  onNavigateAnalysis: (path: string) => void
  onPause: () => void
  onResume: () => void
  onResumeEditOpen: () => void
  onReveal: () => void
  onRerun: () => void
}

export function JobDetailHeader({
  actionError,
  busy,
  compareIds,
  data,
  isLive,
  isPaused,
  isResumable,
  isTerminal,
  jobId,
  showCompareJumpButton,
  streamStatus,
  onArchiveOpen,
  onCancel,
  onCloneOpen,
  onKillOpen,
  onNavigateAnalysis,
  onPause,
  onResume,
  onResumeEditOpen,
  onReveal,
  onRerun,
}: JobDetailHeaderProps) {
  return (
    <header className="space-y-3 border-b border-border/60 px-4 pb-4 pt-12 md:flex md:items-start md:gap-4 md:space-y-0 md:px-7 md:py-5">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          {data && <StateBadge state={data.state} paused={isPaused} />}
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            {streamStatus === "open"
              ? "实时已连接"
              : streamStatus === "closed"
                ? "已断开"
                : "等待中"}
          </span>
        </div>
        <div className="text-base font-semibold tracking-tight font-mono truncate">
          {jobId}
        </div>
        {data && (
          <PathDisplay
            path={data.workspace}
            tailSegments={3}
            block
            className="text-xs text-muted-foreground mt-1"
          />
        )}
        {actionError && (
          <div className="text-[11px] text-destructive mt-2 break-all">
            {actionError}
          </div>
        )}
      </div>
      <div className="no-scrollbar -mx-4 flex items-center gap-2 overflow-x-auto px-4 pb-0.5 md:mx-0 md:shrink-0 md:overflow-visible md:px-0">
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            onNavigateAnalysis(
              showCompareJumpButton
                ? `/analysis/compare?ids=${compareIds.join(",")}`
                : `/analysis/${jobId}`,
            )
          }
          title={
            showCompareJumpButton
              ? "在分析工作台中对比所选任务"
              : "打开训练分析工作台"
          }
          className="shrink-0"
        >
          <BarChart3 className="size-3" />{" "}
          <span>{showCompareJumpButton ? "对比分析" : "深入分析"}</span>
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={onReveal}
          disabled={!data || busy !== null}
          title="在文件管理器中打开工作区"
          aria-label="在文件管理器中打开工作区"
          className="shrink-0"
        >
          {busy === "reveal" ? (
            <Spinner className="size-3" />
          ) : (
            <FolderOpen className="size-3" />
          )}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={onRerun}
          disabled={!data || busy !== null}
          className="shrink-0"
        >
          {busy === "rerun" ? (
            <Spinner className="size-3" />
          ) : (
            <RefreshCw className="size-3" />
          )}{" "}
          <span>再次运行</span>
        </Button>
        {isTerminal && (
          <Button
            variant="outline"
            size="sm"
            onClick={onCloneOpen}
            disabled={busy !== null || !data?.config_snapshot}
            title="从某个 saved state 派生新任务（保留 optimizer / lr 进度，不影响原任务）"
            className="shrink-0"
          >
            <GitBranch className="size-3" /> <span>派生</span>
          </Button>
        )}
        {isTerminal && (
          <Button
            variant="outline"
            size="sm"
            onClick={onArchiveOpen}
            disabled={busy !== null}
            className="shrink-0"
          >
            {busy === "archive" ? (
              <Spinner className="size-3" />
            ) : (
              <Archive className="size-3" />
            )}{" "}
            <span>归档</span>
          </Button>
        )}
        {isResumable && (
          <>
            <Button
              variant="default"
              size="sm"
              onClick={onResume}
              disabled={busy !== null}
              title="从最新 state + safetensors 续训（保留 optimizer / lr 进度）"
              className="shrink-0"
            >
              {busy === "resume" ? (
                <Spinner className="size-3" />
              ) : (
                <Play className="size-3" />
              )}{" "}
              <span>{isPaused ? "继续训练" : "恢复训练"}</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onResumeEditOpen}
              disabled={busy !== null || !data?.config_snapshot}
              title="先编辑 lr / dropTokens / 数据集等再续训（权重相关字段会被锁）"
              className="shrink-0"
            >
              <Pencil className="size-3" /> <span>编辑</span>
            </Button>
          </>
        )}
        {isLive && (
          <>
            {data?.state === "running" ? (
              <Button
                variant="outline"
                size="sm"
                onClick={onPause}
                disabled={busy !== null}
                title="发送 SIGINT，等待训练写出最新 state 后停止；之后点「继续训练」从此处续"
                className="shrink-0"
              >
                {busy === "pause" ? (
                  <Spinner className="size-3" />
                ) : (
                  <Pause className="size-3" />
                )}{" "}
                <span>暂停</span>
              </Button>
            ) : null}
            <Button variant="destructive" size="sm" onClick={onCancel} className="shrink-0">
              <Square className="size-3" /> <span>取消</span>
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={onKillOpen}
              title="强制 SIGKILL 进程组（用于卡死的训练任务）"
              disabled={busy !== null || !data?.pid}
              className="shrink-0"
            >
              {busy === "kill" ? (
                <Spinner className="size-3" />
              ) : (
                <Skull className="size-3" />
              )}{" "}
              <span>强制终止</span>
            </Button>
          </>
        )}
        {!isLive && data?.state === "interrupted" && data.pid && (
          <Button
            variant="outline"
            size="sm"
            onClick={onKillOpen}
            title="任务标记为 interrupted 但 PID 仍可能存活，可强制清理"
            disabled={busy !== null}
            className="shrink-0"
          >
            <Skull className="size-3" /> <span>强制终止</span>
          </Button>
        )}
      </div>
    </header>
  )
}
