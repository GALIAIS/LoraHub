/**
 * Per-route ErrorBoundary so a crash inside one page doesn't take
 * down the entire workbench.
 *
 * React 19 still doesn't ship a hooks-based boundary, so this is the
 * standard class component — minimum-viable, no logging side-effects
 * (real error reporting belongs in an instrumentation layer, not
 * coupled to UI). Resets via the `resetKey` prop: when the route
 * changes, `<App>` rotates the key and the boundary unmounts itself.
 */
import { Component, type ErrorInfo, type ReactNode } from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Props {
  /** Re-mount the boundary (and clear the captured error) when this changes. */
  resetKey?: string | number
  /** Override the default friendly chrome. */
  fallback?: (
    error: Error,
    reset: () => void,
  ) => ReactNode
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // We deliberately don't ship to a remote logger here. The dev
    // console + an optional Sentry-style hook should live somewhere
    // else; this class' single responsibility is the user-visible
    // recovery surface.
    if (typeof console !== "undefined" && console.error) {
      console.error("ErrorBoundary caught:", error, info.componentStack)
    }
  }

  componentDidUpdate(prev: Props) {
    // When the parent rotates `resetKey` (e.g. on route change), drop
    // any captured error so the user can navigate forward without
    // staring at the previous failure.
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  reset = () => this.setState({ error: null })

  render() {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset)
      }
      return <DefaultFallback error={this.state.error} reset={this.reset} />
    }
    return this.props.children
  }
}

function DefaultFallback({ error, reset }: { error: Error; reset: () => void }) {
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
          的「重试」让 LoraHub 重新渲染当前路由,或切到别的页面。
        </p>
        <pre className="rounded-[4px] border border-border/60 bg-muted/40 px-2 py-1.5 text-[11px] font-mono text-foreground/85 max-h-[180px] overflow-auto">
          {error.message || String(error)}
        </pre>
        <div className="flex justify-end">
          <Button size="sm" onClick={reset} className="gap-1.5">
            <RefreshCw className="size-3" /> 重试
          </Button>
        </div>
      </div>
    </div>
  )
}
