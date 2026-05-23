/**
 * Per-route ErrorBoundary so a crash inside one page doesn't take
 * down the entire workbench.
 *
 * React 19 still doesn't ship a hooks-based boundary, so this is the
 * standard class component. ``componentDidCatch`` also fires the
 * client-side reporter so the failure shows up in Settings → 错误上报
 * — without that, only the per-route fallback would survive the crash
 * and a refresh would lose the trace forever. Resets via the
 * ``resetKey`` prop: when the route changes, `<App>` rotates the key
 * and the boundary unmounts itself.
 */
import { Component, type ErrorInfo, type ReactNode } from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { reportError } from "@/lib/error-reporter"

interface Props {
  /** Re-mount the boundary (and clear the captured error) when this changes. */
  resetKey?: string | number
  /** Override the default friendly chrome. */
  fallback?: (
    error: Error,
    reset: () => void,
    reportId: string | null,
  ) => ReactNode
  /** Reporter source label so root vs route boundaries are distinguishable. */
  reporterSource?: string
  children: ReactNode
}

interface State {
  error: Error | null
  reportId: string | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, reportId: null }

  static getDerivedStateFromError(error: Error): Pick<State, "error"> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (typeof console !== "undefined" && console.error) {
      console.error("ErrorBoundary caught:", error, info.componentStack)
    }
    void reportError({
      severity: "error",
      source: this.props.reporterSource ?? "frontend.render",
      category: "boundary",
      title: error.name || "render error",
      message: error.message || String(error),
      stack: error.stack ?? null,
      context: {
        componentStack: info.componentStack,
        href: typeof window !== "undefined" ? window.location.href : null,
      },
      requestPath:
        typeof window !== "undefined" ? window.location.pathname : null,
    }).then((id) => {
      if (id && this.state.error) this.setState({ reportId: id })
    })
  }

  componentDidUpdate(prev: Props) {
    // When the parent rotates `resetKey` (e.g. on route change), drop
    // any captured error so the user can navigate forward without
    // staring at the previous failure.
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null, reportId: null })
    }
  }

  reset = () => this.setState({ error: null, reportId: null })

  render() {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset, this.state.reportId)
      }
      return (
        <DefaultFallback
          error={this.state.error}
          reset={this.reset}
          reportId={this.state.reportId}
        />
      )
    }
    return this.props.children
  }
}

function DefaultFallback({
  error,
  reset,
  reportId,
}: {
  error: Error
  reset: () => void
  reportId: string | null
}) {
  return (
    <div className="h-full w-full flex items-center justify-center px-6">
      <div className="max-w-lg w-full rounded-[6px] border border-destructive/40 bg-destructive/5 px-5 py-4 space-y-3">
        <div className="flex items-center gap-2 text-destructive">
          <AlertTriangle className="size-4" />
          <h2 className="text-[13px] font-semibold tracking-tight">
            页面渲染异常
          </h2>
        </div>
        <p className="text-[12px] text-muted-foreground leading-relaxed">
          这一节子树抛出了未捕获的错误。仪表盘其余部分仍然可用,你可以点击下方
          的「重试」让 LoraHub 重新渲染当前路由,或切到别的页面。完整堆栈已记录在
          「设置 → 错误上报」中。
        </p>
        <pre className="rounded-[4px] border border-border/60 bg-muted/40 px-2 py-1.5 text-[11px] font-mono text-foreground/85 max-h-[180px] overflow-auto">
          {error.message || String(error)}
        </pre>
        {reportId && (
          <div className="text-[11px] text-muted-foreground font-mono">
            错误 ID: <span className="text-foreground">{reportId}</span>
          </div>
        )}
        <div className="flex justify-end gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => window.location.assign("/settings")}
          >
            打开错误上报
          </Button>
          <Button size="sm" onClick={reset} className="gap-1.5">
            <RefreshCw className="size-3" /> 重试
          </Button>
        </div>
      </div>
    </div>
  )
}
