/**
 * Stage stepper for the image studio workbench.
 *
 * Five stages (intake / audit / curate / annotate / ship) representing
 * the dataset preparation workflow. Each stage shows a status icon
 * (idle / in-progress / done / warn / error) so the user gets a
 * pipeline-shaped overview of what's left to do.
 */
import { Check, AlertTriangle, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

export type StageId = "intake" | "audit" | "curate" | "annotate" | "ship"
export type StageStatus = "idle" | "active" | "done" | "warn" | "error"

export interface StageInfo {
  id: StageId
  label: string
  hint?: string
  status: StageStatus
}

export const DEFAULT_STAGES: readonly StageInfo[] = [
  { id: "intake",   label: "导入", hint: "拖入 / 复制 / 路径吸附", status: "idle" },
  { id: "audit",    label: "审计", hint: "扫描数据集健康", status: "idle" },
  { id: "curate",   label: "整理", hint: "裁剪 / 删除 / 重采样", status: "idle" },
  { id: "annotate", label: "标注", hint: "WD14 / VLM / 标签管理", status: "idle" },
  { id: "ship",     label: "输出", hint: "训练 / 导出 / 同步", status: "idle" },
] as const

interface Props {
  stages: readonly StageInfo[]
  active: StageId
  onSelect: (id: StageId) => void
}

export function StageStepper({ stages, active, onSelect }: Props) {
  return (
    <nav
      role="tablist"
      aria-label="数据集处理阶段"
      className="flex items-stretch border-b border-border/60 bg-background"
    >
      {stages.map((s, i) => {
        const isActive = s.id === active
        return (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onSelect(s.id)}
            className={cn(
              "relative flex flex-1 items-center gap-2 px-4 py-2.5 text-left",
              "border-r border-border/60 last:border-r-0",
              "transition-colors",
              isActive
                ? "bg-muted/50"
                : "hover:bg-muted/30 text-muted-foreground hover:text-foreground",
            )}
          >
            <span
              className={cn(
                "inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[11px] font-medium",
                statusBadgeClass(s.status, isActive),
              )}
            >
              <StageStatusIcon status={s.status} index={i} />
            </span>
            <span className="flex flex-col min-w-0">
              <span
                className={cn(
                  "text-sm leading-tight",
                  isActive ? "text-foreground font-medium" : "",
                )}
              >
                {s.label}
              </span>
              {s.hint && (
                <span className="truncate text-[10px] text-muted-foreground/80">
                  {s.hint}
                </span>
              )}
            </span>
            {isActive && (
              <span
                aria-hidden
                className="absolute bottom-0 left-0 right-0 h-[2px] bg-primary"
              />
            )}
          </button>
        )
      })}
    </nav>
  )
}

function statusBadgeClass(status: StageStatus, isActive: boolean): string {
  if (status === "done") return "border-emerald-600/60 bg-emerald-600/15 text-emerald-700 dark:text-emerald-400"
  if (status === "warn") return "border-amber-600/60 bg-amber-600/15 text-amber-700 dark:text-amber-400"
  if (status === "error") return "border-red-600/60 bg-red-600/15 text-red-700 dark:text-red-400"
  if (status === "active") return "border-primary bg-primary/10 text-primary"
  return isActive
    ? "border-primary/40 text-primary"
    : "border-border text-muted-foreground"
}

function StageStatusIcon({ status, index }: { status: StageStatus; index: number }) {
  if (status === "done") return <Check className="size-3.5" />
  if (status === "warn") return <AlertTriangle className="size-3.5" />
  if (status === "error") return <AlertTriangle className="size-3.5" />
  if (status === "active") return <Loader2 className="size-3.5 animate-spin" />
  return <span className="text-[10px] tabular-nums">{index + 1}</span>
}
