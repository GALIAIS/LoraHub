import { Suspense, useCallback, useEffect, useState } from "react"
import { NavLink, Outlet, useLocation } from "react-router-dom"
import { useQueryClient } from "@tanstack/react-query"
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
  Package,
  Palette,
  Settings,
  SlidersHorizontal,
  Sun,
  TerminalSquare,
} from "lucide-react"
import { preloadAppRoute, type AppRouteModuleKey } from "@/app/route-modules"
import { api } from "@/lib/api"
import { ErrorBoundary } from "@/components/error-boundary"
import { GlobalStatusBar } from "@/components/global-status-bar"
import { useVersionInfo } from "@/hooks/use-version-info"
import { useAnimeEnter } from "@/hooks/use-anime-enter"
import { useAnimeThemeTransition } from "@/hooks/use-anime-theme-transition"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { cn } from "@/lib/utils"

const NAV: Array<{
  to: string
  label: string
  icon: typeof Activity
  routeKey: AppRouteModuleKey
  group: "workspace" | "tools" | "system"
}> = [
  { to: "/", label: "数据面板", icon: Activity, routeKey: "dashboard", group: "workspace" },
  { to: "/jobs", label: "训练任务", icon: ListTree, routeKey: "jobs", group: "workspace" },
  { to: "/analysis", label: "训练分析", icon: BarChart3, routeKey: "analysis", group: "workspace" },
  { to: "/configs", label: "训练配置", icon: Layers, routeKey: "configs", group: "workspace" },
  { to: "/sweeps", label: "参数搜索", icon: SlidersHorizontal, routeKey: "sweeps", group: "workspace" },
  { to: "/terminal", label: "终端", icon: TerminalSquare, routeKey: "terminal", group: "tools" },
  { to: "/datasets", label: "数据集", icon: Database, routeKey: "datasets", group: "tools" },
  { to: "/image-studio", label: "图像工作台", icon: Palette, routeKey: "image-studio", group: "tools" },
  { to: "/gallery", label: "样图画廊", icon: Images, routeKey: "gallery", group: "tools" },
  { to: "/artifacts", label: "产物归档", icon: Package, routeKey: "artifacts", group: "tools" },
  { to: "/settings", label: "设置", icon: Settings, routeKey: "settings", group: "system" },
  { to: "/about", label: "关于", icon: Info, routeKey: "about", group: "system" },
]

const NAV_GROUPS: Array<{ key: string; label: string }> = [
  { key: "workspace", label: "工作区" },
  { key: "tools", label: "工具" },
  { key: "system", label: "系统" },
]

const MOBILE_NAV = [
  "/",
  "/jobs",
  "/analysis",
  "/configs",
  "/image-studio",
] as const

type ThemeMode = "light" | "dark" | "system"
type StyleMode = "shiro" | "polar"

const THEME_MODE_KEY = "lorahub.theme.mode"
const STYLE_MODE_KEY = "lorahub.ui.style"

export default function App() {
  const location = useLocation()
  const queryClient = useQueryClient()
  const [mode, setMode] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "system"
    const stored = window.localStorage.getItem(THEME_MODE_KEY)
    return stored === "light" || stored === "dark" || stored === "system" ? stored : "system"
  })
  const [styleMode, setStyleMode] = useState<StyleMode>(() => {
    if (typeof window === "undefined") return "shiro"
    const stored = window.localStorage.getItem(STYLE_MODE_KEY)
    return stored === "polar" || stored === "shiro" ? stored : "shiro"
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    const root = document.documentElement
    const media = window.matchMedia("(prefers-color-scheme: dark)")

    const apply = () => {
      const dark = mode === "dark" || (mode === "system" && media.matches)
      root.classList.toggle("dark", dark)
      root.dataset.themeMode = mode
      root.removeAttribute("data-theme-accent")
      window.localStorage.setItem(THEME_MODE_KEY, mode)
      window.localStorage.removeItem("lorahub.theme.accent")
    }

    apply()
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [mode])

  useEffect(() => {
    if (typeof window === "undefined") return
    const root = document.documentElement
    root.dataset.uiStyle = styleMode
    window.localStorage.setItem(STYLE_MODE_KEY, styleMode)
  }, [styleMode])

  useAnimeThemeTransition([mode, styleMode])

  // Eagerly warm every route chunk during browser idle time so the
  // first click on a nav item never pays a network round-trip. We
  // also kick a queueMicrotask fallback for browsers without
  // requestIdleCallback (Safari).
  useEffect(() => {
    if (typeof window === "undefined") return
    const keys = NAV.map((n) => n.routeKey)
    let cancelled = false

    const warm = () => {
      if (cancelled) return
      for (const key of keys) {
        void preloadAppRoute(key)
      }
      // Prefetch the workbench-level settings so pages that need
      // `default_backend` (e.g. ConfigsPage's backend filter) don't
      // flash an unfiltered list before the response arrives.
      void queryClient.prefetchQuery({
        queryKey: ["settings"],
        queryFn: api.getSettings,
        staleTime: 60_000,
      })
    }

    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number
      cancelIdleCallback?: (id: number) => void
    }

    let idleId: number | undefined
    let timerId: number | undefined
    if (typeof w.requestIdleCallback === "function") {
      idleId = w.requestIdleCallback(warm, { timeout: 2000 })
    } else {
      timerId = window.setTimeout(warm, 600)
    }

    return () => {
      cancelled = true
      if (idleId !== undefined && typeof w.cancelIdleCallback === "function") {
        w.cancelIdleCallback(idleId)
      }
      if (timerId !== undefined) window.clearTimeout(timerId)
    }
  }, [])

  const prefetchRoute = useCallback((routeKey: AppRouteModuleKey) => {
    void preloadAppRoute(routeKey)
  }, [])

  // Theme-change handler. Uses the View Transitions API for a radial
  // reveal centered on the click, falling back to the css fade class
  // (see index.css) on browsers that don't support it.
  const runThemeChange = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>, apply: () => void) => {
      const docAny = document as Document & {
        startViewTransition?: (cb: () => void) => { finished: Promise<void> }
      }
      if (typeof docAny.startViewTransition !== "function") {
        apply()
        return
      }
      const root = document.documentElement
      const rect = event.currentTarget.getBoundingClientRect()
      const x = rect.left + rect.width / 2
      const y = rect.top + rect.height / 2
      root.style.setProperty("--theme-origin-x", `${x}px`)
      root.style.setProperty("--theme-origin-y", `${y}px`)
      root.dataset.viewTransitionInProgress = "true"
      const transition = docAny.startViewTransition(() => {
        apply()
      })
      transition.finished.finally(() => {
        delete root.dataset.viewTransitionInProgress
      })
    },
    [],
  )

  const isRouteActive = (href: string) =>
    href === "/"
      ? location.pathname === "/"
      : location.pathname === href || location.pathname.startsWith(`${href}/`)

  const resolvedTitle =
    NAV.find((n) => isRouteActive(n.to))?.label ?? "LoraHub"
  const pageEnterRef = useAnimeEnter<HTMLDivElement>([location.pathname])

  return (
    <SidebarProvider className="shiro-shell-grid">
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

      {/* --- Sidebar --- */}
      <Sidebar variant="inset">
        <SidebarHeader className="border-b border-sidebar-border/70 px-4 py-3">
          <div className="px-1">
            <div className="text-sm font-semibold tracking-tight">LoraHub</div>
            <SidebarVersionStack />
          </div>
        </SidebarHeader>

        <SidebarContent className="px-3 py-4">
          {NAV_GROUPS.map((group, index) => (
            <SidebarGroup key={group.key} className={index > 0 ? "mt-5" : undefined}>
              <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {NAV.filter((n) => n.group === group.key).map((item) => (
                    <SidebarMenuItem key={item.to}>
                      <SidebarMenuButton
                        render={
                          <NavLink
                            to={item.to}
                            end={item.to === "/"}
                            onPointerDown={() => prefetchRoute(item.routeKey)}
                          />
                        }
                        isActive={isRouteActive(item.to)}
                        tooltip={item.label}
                        onMouseEnter={() => prefetchRoute(item.routeKey)}
                        onFocus={() => prefetchRoute(item.routeKey)}
                      >
                        <item.icon />
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          ))}
        </SidebarContent>

        <SidebarFooter className="border-t border-sidebar-border/70 px-4 py-4">
          <div className="space-y-1.5 px-1">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
              <Monitor className="size-3" /> 外观
            </div>
            <div className="grid grid-cols-3 gap-1">
              {([
                { value: "light" as const, icon: Sun, label: "浅色" },
                { value: "dark" as const, icon: Moon, label: "深色" },
                { value: "system" as const, icon: Monitor, label: "系统" },
              ]).map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={(e) => runThemeChange(e, () => setMode(value))}
                  className={cn(
                    "h-7 rounded-[6px] border text-[11px] inline-flex items-center justify-center gap-1 transition-colors duration-150",
                    mode === value
                      ? "border-sidebar-primary/50 bg-sidebar-accent text-sidebar-accent-foreground shadow-[0_1px_0_rgba(255,255,255,0.08)_inset]"
                      : "border-sidebar-border/60 text-muted-foreground hover:border-sidebar-border hover:bg-sidebar-accent/60 hover:text-foreground",
                  )}
                  title={label}
                >
                  <Icon className="size-3" />
                </button>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-1 pt-1">
              {([
                { value: "shiro" as const, label: "Shiro" },
                { value: "polar" as const, label: "Polar" },
              ]).map(({ value, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={(e) => runThemeChange(e, () => setStyleMode(value))}
                  className={cn(
                    "h-7 rounded-[6px] border px-2 text-[11px] font-medium transition-colors duration-150",
                    styleMode === value
                      ? "border-sidebar-primary/50 bg-sidebar-accent text-sidebar-accent-foreground shadow-[0_1px_0_rgba(255,255,255,0.08)_inset]"
                      : "border-sidebar-border/60 text-muted-foreground hover:border-sidebar-border hover:bg-sidebar-accent/60 hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </SidebarFooter>
      </Sidebar>

      {/* --- Main content --- */}
      <SidebarInset className="shiro-page-canvas h-[100dvh] overflow-hidden">
        <div className="relative flex h-full flex-col overflow-hidden">
          {/* Header */}
          <header className="shrink-0 z-30">
            <div className="shiro-toolbar px-3 py-3 md:px-4">
              <div className="flex items-center gap-3">
                <SidebarTrigger />
                <div className="hidden h-8 w-px bg-border/70 md:block" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-base font-semibold tracking-[-0.01em]">
                    {resolvedTitle}
                  </div>
                </div>
              </div>
            </div>
            <GlobalStatusBar />
          </header>

          {/* Page content */}
          <div className="flex w-full min-w-0 min-h-0 flex-1 flex-col overflow-hidden pb-[calc(4.25rem+env(safe-area-inset-bottom))] md:pb-0">
            <Suspense
              fallback={
                <div
                  role="status"
                  aria-live="polite"
                  className="h-full w-full bg-muted/30 shiro-loading-pulse"
                />
              }
            >
              <ErrorBoundary resetKey={location.pathname}>
                <div ref={pageEnterRef} className="flex-1 min-h-0 flex flex-col">
                  <Outlet />
                </div>
              </ErrorBoundary>
            </Suspense>
          </div>
          <MobileBottomNav isRouteActive={isRouteActive} prefetchRoute={prefetchRoute} />
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

function MobileBottomNav({
  isRouteActive,
  prefetchRoute,
}: {
  isRouteActive: (href: string) => boolean
  prefetchRoute: (routeKey: AppRouteModuleKey) => void
}) {
  const items = MOBILE_NAV
    .map((href) => NAV.find((item) => item.to === href))
    .filter((item): item is NonNullable<typeof item> => Boolean(item))

  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-40 border-t border-border/70 bg-background/92 px-2 pb-[calc(0.5rem+env(safe-area-inset-bottom))] pt-2 shadow-[0_-18px_40px_-30px_rgba(15,23,42,0.45)] backdrop-blur-xl md:hidden"
      aria-label="移动端主导航"
    >
      <div className="grid grid-cols-5 gap-1">
        {items.map((item) => {
          const active = isRouteActive(item.to)
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onPointerDown={() => prefetchRoute(item.routeKey)}
              onMouseEnter={() => prefetchRoute(item.routeKey)}
              onFocus={() => prefetchRoute(item.routeKey)}
              className={cn(
                "flex min-w-0 flex-col items-center justify-center gap-1 rounded-[8px] px-1 py-1.5 text-[10px] font-medium transition",
                active
                  ? "bg-primary/10 text-foreground ring-1 ring-primary/20"
                  : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" />
              <span className="max-w-full truncate leading-none">{item.label}</span>
            </NavLink>
          )
        })}
      </div>
    </nav>
  )
}

/**
 * Version chips under the sidebar subtitle.
 *
 * Layout: two stacked rows: Frontend on top, Backend below, sitting
 * under the "LoRA 训练工作台" tagline. The vertical stack reads cleanly
 * even when the two version strings disagree by length, and keeps the
 * existing kicker / subtitle / chips rhythm intact.
 *
 * Both rows turn amber together when frontend and backend resolve to
 * *different commits*. We compare commit shas first (the canonical
 * answer); when shas match we display the same canonical string on both
 * sides, since git-describe ("last tag") and hatch-vcs ("next tag")
 * disagree on the textual base of every untagged commit and showing both
 * raw views would falsely suggest drift.
 *
 * Clicking jumps to the About page where the long-form mismatch card
 * spells out the recovery commands (`lorahub manage build`, etc.).
 */
function SidebarVersionStack() {
  const {
    frontendDisplay,
    backendDisplay,
    mismatch,
    loading,
  } = useVersionInfo()
  const labelTone = mismatch
    ? "text-amber-700 dark:text-amber-400"
    : "text-sidebar-foreground/55"
  const valueTone = mismatch
    ? "text-amber-700 dark:text-amber-400"
    : "text-sidebar-foreground/80"
  const title = mismatch
    ? `前端 ${frontendDisplay} 与后端 ${backendDisplay} 来自不同 commit。请运行 \`scripts/run.bat dev\` 或 \`lorahub manage build\` 重建前端。点击查看详情。`
    : loading
      ? "正在读取后端版本…"
      : `前端 ${frontendDisplay} · 后端 ${backendDisplay} (同一 commit)`
  return (
    <NavLink
      to="/about"
      title={title}
      className={cn(
        "mt-2 flex flex-col gap-0.5 font-mono tabular-nums tracking-tight transition-colors",
        "text-[10px] leading-snug",
        "hover:text-sidebar-accent-foreground",
      )}
    >
      <span className="inline-flex items-center gap-1">
        <span className={cn("w-[3.5rem]", labelTone)}>Frontend</span>
        <span className={cn("font-medium", valueTone)}>{frontendDisplay}</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <span className={cn("w-[3.5rem]", labelTone)}>Backend</span>
        <span className={cn("font-medium", valueTone)}>
          {loading && backendDisplay === "?" ? "…" : backendDisplay}
        </span>
      </span>
    </NavLink>
  )
}
