import { Suspense, useEffect, useState } from "react"
import { NavLink, Outlet, useLocation } from "react-router-dom"
import { Toaster } from "sonner"
import {
  Activity,
  BarChart3,
  Database,
  Images,
  Info,
  Layers,
  ListTree,
  Monitor,
  Moon,
  Palette,
  Settings,
  SlidersHorizontal,
  Sun,
} from "lucide-react"
import { ErrorBoundary } from "@/components/error-boundary"
import { GlobalStatusBar } from "@/components/global-status-bar"
import { cn } from "@/lib/utils"

const NAV = [
  { to: "/", label: "数据面板", icon: Activity },
  { to: "/jobs", label: "训练任务", icon: ListTree },
  { to: "/analysis", label: "训练分析", icon: BarChart3 },
  { to: "/sweeps", label: "超参 sweep", icon: SlidersHorizontal },
  { to: "/configs", label: "训练配置", icon: Layers },
  { to: "/datasets", label: "数据集", icon: Database },
  { to: "/image-studio", label: "图像工作台", icon: Palette },
  { to: "/gallery", label: "样图画廊", icon: Images },
  { to: "/settings", label: "设置", icon: Settings },
  { to: "/about", label: "关于", icon: Info },
]

type ThemeMode = "light" | "dark" | "system"
type AccentTheme = "slate" | "cyan" | "amber" | "rose"

const THEME_MODE_KEY = "lorahub.theme.mode"
const ACCENT_KEY = "lorahub.theme.accent"
const ACCENTS: Array<{ value: AccentTheme; label: string }> = [
  { value: "slate", label: "石墨" },
  { value: "cyan", label: "青蓝" },
  { value: "amber", label: "琥珀" },
  { value: "rose", label: "蔷薇" },
]

export default function App() {
  // Route key — used both as React Suspense / ErrorBoundary reset key
  // so both unwedge themselves on navigation.
  const location = useLocation()
  const [mode, setMode] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "system"
    const stored = window.localStorage.getItem(THEME_MODE_KEY)
    return stored === "light" || stored === "dark" || stored === "system" ? stored : "system"
  })
  const [accent, setAccent] = useState<AccentTheme>(() => {
    if (typeof window === "undefined") return "slate"
    const stored = window.localStorage.getItem(ACCENT_KEY)
    return stored === "cyan" || stored === "amber" || stored === "rose" ? stored : "slate"
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    const root = document.documentElement
    const media = window.matchMedia("(prefers-color-scheme: dark)")

    const apply = () => {
      const dark = mode === "dark" || (mode === "system" && media.matches)
      root.classList.toggle("dark", dark)
      root.dataset.themeMode = mode
      root.dataset.themeAccent = accent
      window.localStorage.setItem(THEME_MODE_KEY, mode)
      window.localStorage.setItem(ACCENT_KEY, accent)
    }

    apply()
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [mode, accent])

  useEffect(() => {
    // Block right-click only. We used to also blanket-block `copy` /
    // `cut`, but that swallowed Ctrl-C on legitimate things the user
    // wanted to copy — stack traces in error chips, workspace paths,
    // command-line strings, model names. Selection is governed by the
    // `user-select: none` declared on the body in index.css; any
    // element that wants user copy support opts in with the
    // `select-text` utility class instead.
    //
    // NOTE: do NOT add `selectstart` here either — it fires whenever
    // the mouse moves a couple of pixels during a click, and once we
    // preventDefault it the browser treats the gesture as a drag and
    // silently swallows the subsequent `click` event. That broke
    // sidebar navigation when users clicked too quickly.
    const prevent = (e: Event) => e.preventDefault()
    document.addEventListener("contextmenu", prevent)
    return () => {
      document.removeEventListener("contextmenu", prevent)
    }
  }, [])

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      {/* Global toast surface. `richColors` picks up our destructive /
          success tokens automatically; we pin position to bottom-right
          so the running tail of events at the top of jobs / dashboard
          isn't covered. `theme="system"` follows the user's `data-
          themeMode` toggle on <html>. */}
      <Toaster
        position="bottom-right"
        richColors
        closeButton
        theme="system"
        toastOptions={{
          classNames: {
            toast:
              "border-border/60 bg-background text-foreground shadow-[var(--panel-shadow)]",
            description: "text-muted-foreground",
          },
        }}
      />
      <aside
        aria-label="主导航"
        className="w-56 shrink-0 border-r border-sidebar-border bg-sidebar/95 backdrop-blur px-3 py-5 flex flex-col gap-1 overflow-y-auto"
      >
        <div className="px-2 mb-5 flex items-center gap-2">
          <div className="size-8 rounded-[6px] bg-primary text-primary-foreground grid place-items-center font-semibold tracking-tight text-sm">
            L
          </div>
          <div className="flex-1">
            <div className="text-[13px] font-semibold leading-tight">LoraHub</div>
            <div className="text-[11px] text-muted-foreground">训练工作台</div>
          </div>
        </div>

        <nav aria-label="工作台" className="flex flex-col gap-0.5">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-2 rounded-[2px] px-2.5 py-1.5 text-[13px] transition-colors",
                  "border border-transparent",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground border-sidebar-border/60 shadow-[0_1px_0_rgba(255,255,255,0.42)_inset]"
                    : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                )
              }
            >
              {/* react-router's NavLink already sets aria-current="page"
                  on the active link automatically — no manual prop
                  needed. The icon is decorative; aria-hidden keeps it
                  out of the accessibility tree so the link's accessible
                  name comes from the label text alone. */}
              <Icon className="size-3.5" aria-hidden />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto space-y-3 px-2 pt-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
              <Monitor className="size-3" /> 外观
            </div>
            <div className="grid grid-cols-3 gap-1">
              {[
                { value: "light" as const, icon: Sun, label: "浅色" },
                { value: "dark" as const, icon: Moon, label: "深色" },
                { value: "system" as const, icon: Monitor, label: "系统" },
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setMode(value)}
                  className={cn(
                    "h-7 rounded-[2px] border text-[11px] inline-flex items-center justify-center gap-1 transition-colors",
                    mode === value
                      ? "border-sidebar-primary/50 bg-sidebar-accent text-sidebar-accent-foreground"
                      : "border-sidebar-border/60 text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                  )}
                  title={label}
                >
                  <Icon className="size-3" />
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
              <Palette className="size-3" /> 主题色
            </div>
            <div className="grid grid-cols-2 gap-1">
              {ACCENTS.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  onClick={() => setAccent(item.value)}
                  className={cn(
                    "h-7 rounded-[2px] border px-2 text-[11px] transition-colors",
                    accent === item.value
                      ? "border-sidebar-primary/50 bg-sidebar-accent text-sidebar-accent-foreground"
                      : "border-sidebar-border/60 text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                  )}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div
            className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70 font-mono"
            title="Resolved at build time from git describe; see vite.config.ts"
          >
            v{__APP_VERSION__}
          </div>
        </div>
      </aside>

      <main className="flex-1 min-w-0 h-full overflow-hidden flex flex-col">
        <GlobalStatusBar />
        <div className="flex-1 min-h-0 overflow-hidden">
          {/* Suspense lives between the shell chrome and the page body
              so route chunk loads don't briefly unmount the sidebar /
              status bar. The fallback is intentionally a blank surface
              — chunks resolve in <300 ms over a warm cache, faster
              than a spinner is helpful. */}
          <Suspense
            fallback={
              <div
                role="status"
                aria-live="polite"
                className="h-full w-full bg-background/40"
              />
            }
          >
            <ErrorBoundary resetKey={location.pathname}>
              <Outlet />
            </ErrorBoundary>
          </Suspense>
        </div>
      </main>
    </div>
  )
}
