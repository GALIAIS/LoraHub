import { CheckCircle2, XCircle, AlertTriangle, ExternalLink } from "lucide-react"
import type {
  AnyBackendStatus,
  BackendDescriptor,
  KohyaBackendStatus,
  DiffusionPipeBackendStatus,
} from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// Compact status icon used in the header of every backend card. We use three
// tones so a kohya-style "sd-scripts present but no python" is visually
// different from a complete miss.
function StatusIcon({ ok, warn }: { ok: boolean; warn: boolean }) {
  if (ok) return <CheckCircle2 className="size-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
  if (warn) return <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400 shrink-0" />
  return <XCircle className="size-4 text-destructive shrink-0" />
}

function isKohya(s: AnyBackendStatus): s is KohyaBackendStatus {
  return s.id === "kohya"
}

function isDiffusionPipe(s: AnyBackendStatus): s is DiffusionPipeBackendStatus {
  return s.id === "diffusion-pipe"
}

function statusTone(s: AnyBackendStatus) {
  if (s.ready) return { tone: "ready" as const, label: "已就绪" }
  if (isKohya(s) && s.sd_scripts_ok && s.python_ok && !s.requirements_ok) {
    return { tone: "warn" as const, label: "缺少依赖" }
  }
  if (isDiffusionPipe(s) && s.repo_ok && s.python_ok && !s.requirements_ok) {
    return { tone: "warn" as const, label: "缺少依赖" }
  }
  if (isKohya(s) && s.sd_scripts_ok && !s.python_ok) {
    return { tone: "warn" as const, label: "缺少 Python" }
  }
  if (isDiffusionPipe(s) && s.repo_ok && !s.python_ok) {
    return { tone: "warn" as const, label: "缺少 Python" }
  }
  return { tone: "broken" as const, label: "未配置" }
}

function repoPath(s: AnyBackendStatus): string {
  return isKohya(s) ? s.sd_scripts_path : s.repo_path
}

function missing(s: AnyBackendStatus): string[] {
  return isKohya(s) ? s.missing_scripts : s.missing_files
}

export interface BackendStatusCardProps {
  descriptor: BackendDescriptor
  isDefault: boolean
  onMakeDefault?: () => void
  makingDefault?: boolean
  /** When true, hide the "set default" / "open repo" actions. */
  compact?: boolean
}

/**
 * Generic status card for any registered backend. The card is read-only —
 * actual editing of paths happens in the "后端管理" tab; this component is
 * shared by the overview tab and the install tab.
 */
export function BackendStatusCard({
  descriptor,
  isDefault,
  onMakeDefault,
  makingDefault,
  compact,
}: BackendStatusCardProps) {
  const status = descriptor.status
  const { tone, label } = statusTone(status)
  const toneClass = {
    ready: "border-emerald-500/40 bg-emerald-500/5",
    warn: "border-amber-500/40 bg-amber-500/5",
    broken: "border-destructive/40 bg-destructive/5",
  }[tone]
  const warn = tone === "warn"
  const missingFiles = missing(status)

  return (
    <div
      className={cn(
        "rounded-[6px] border px-4 py-3 shadow-[var(--panel-shadow)] flex flex-col gap-2.5",
        toneClass,
      )}
    >
      <div className="flex items-start gap-3">
        <StatusIcon ok={status.ready} warn={warn} />
        <div className="flex-1 min-w-0 space-y-0.5">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold tracking-tight">
              {descriptor.name}
            </span>
            {isDefault && (
              <Badge variant="default" className="h-4 text-[9px] tracking-[0.18em]">
                默认
              </Badge>
            )}
            <span
              className={cn(
                "text-[10px] uppercase tracking-[0.18em] font-mono px-1.5 py-0.5 rounded-[2px] border",
                tone === "ready" &&
                  "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/5",
                tone === "warn" &&
                  "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/5",
                tone === "broken" &&
                  "text-destructive border-destructive/40 bg-destructive/5",
              )}
            >
              {label}
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground/90 leading-snug">
            {descriptor.description}
          </p>
        </div>
        {!compact && (
          <div className="flex items-center gap-1.5 shrink-0">
            {!isDefault && onMakeDefault && (
              <Button
                size="sm"
                variant="outline"
                onClick={onMakeDefault}
                disabled={makingDefault}
              >
                设为默认
              </Button>
            )}
            <a
              href={descriptor.repo_url}
              target="_blank"
              rel="noreferrer"
              title="打开仓库"
              className="inline-flex items-center gap-1 h-8 px-2.5 text-[0.82rem] rounded-[2px] border border-transparent text-muted-foreground hover:bg-muted/72 hover:text-foreground transition-colors"
            >
              <ExternalLink className="size-3.5" />
              仓库
            </a>
          </div>
        )}
      </div>

      <dl className="text-xs grid grid-cols-[5.5rem_1fr] gap-x-3 gap-y-0.5 font-mono">
        <dt className="text-muted-foreground">路径</dt>
        <dd className="truncate" title={repoPath(status)}>
          {repoPath(status)}
        </dd>
        <dt className="text-muted-foreground">python</dt>
        <dd className="truncate" title={status.python ?? ""}>
          {status.python ?? "—"}
        </dd>
        <dt className="text-muted-foreground">来源</dt>
        <dd>{status.source}</dd>
      </dl>

      {missingFiles.length > 0 && (
        <div className="text-[11px] text-destructive">
          缺失文件：{missingFiles.join(", ")}
        </div>
      )}

      {status.missing_requirements.length > 0 && (
        <div className="text-[11px] text-amber-600 dark:text-amber-400">
          缺失依赖：{status.missing_requirements.length > 5
            ? `${status.missing_requirements.slice(0, 5).join(", ")} 等 ${status.missing_requirements.length} 项`
            : status.missing_requirements.join(", ")}
        </div>
      )}
    </div>
  )
}
