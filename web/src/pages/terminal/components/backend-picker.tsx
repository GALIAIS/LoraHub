import { Loader2 } from "lucide-react"
import type { TerminalEnvironment } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface BackendPickerProps {
  sessions: TerminalEnvironment[]
  selected: string | null
  loading: boolean
  onChange: (id: string) => void
}

export function BackendPicker({
  sessions,
  selected,
  loading,
  onChange,
}: BackendPickerProps) {
  if (loading) {
    return (
      <div className="text-xs text-muted-foreground flex items-center gap-2">
        <Loader2 className="size-3 animate-spin" /> 正在探测已安装的后端…
      </div>
    )
  }
  if (sessions.length === 0) {
    return (
      <div className="rounded-[6px] border border-dashed border-border/60 bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
        没有可用后端。请到 设置 → 后端管理 安装至少一个后端。
      </div>
    )
  }
  const active = sessions.find((s) => s.backend_id === selected) ?? null
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        {sessions.map((s) => {
          const isActive = s.backend_id === selected
          const disabled = !s.ready
          return (
            <button
              key={s.backend_id}
              type="button"
              onClick={() => !disabled && onChange(s.backend_id)}
              disabled={disabled}
              className={cn(
                "rounded-[5px] border px-3 py-2.5 text-left transition-colors",
                isActive
                  ? "border-primary/60 bg-primary/8"
                  : "border-border/60 hover:bg-muted/50",
                disabled && "opacity-60 cursor-not-allowed",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-sm truncate">{s.name}</span>
                {s.ready ? (
                  <Badge
                    variant="outline"
                    className="rounded-[2px] text-[10px] border-emerald-500/40 text-emerald-700 dark:text-emerald-300"
                  >
                    就绪
                  </Badge>
                ) : (
                  <Badge variant="outline" className="rounded-[2px] text-[10px]">
                    未安装
                  </Badge>
                )}
              </div>
              <div
                className="mt-1 font-mono text-[10.5px] text-muted-foreground truncate"
                title={s.repo_path}
              >
                {s.repo_path || "—"}
              </div>
            </button>
          )
        })}
      </div>

      {active && (
        <div className="rounded-[5px] border border-border/60 bg-muted/20 px-3 py-2 text-[11px] grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-0.5">
          <Field label="Python" value={active.python_path ?? "—"} mono />
          <Field label="venv" value={active.venv_dir ?? "—"} mono />
          <Field label="工作目录" value={active.repo_path} mono />
          <Field
            label="状态"
            value={active.ready ? "ready" : active.venv_detected ? "venv 已找到，依赖未就位" : "未安装"}
          />
        </div>
      )}
    </div>
  )
}

function Field({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-baseline gap-2 min-w-0">
      <span className="text-muted-foreground/70 shrink-0">{label}</span>
      <span
        className={cn("truncate flex-1", mono && "font-mono text-foreground/90")}
        title={value}
      >
        {value}
      </span>
    </div>
  )
}
