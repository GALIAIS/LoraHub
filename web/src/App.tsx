import { Suspense, useCallback, useEffect, useState } from "react"
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
import { preloadAppRoute, type AppRouteModuleKey } from "@/app/route-modules"
import { ErrorBoundary } from "@/components/error-boundary"
import { GlobalStatusBar } from "@/components/global-status-bar"
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
  { to: "/sweeps", label: "超参 sweep", icon: SlidersHorizontal, routeKey: "sweeps", group: "workspace" },
  { to: "/configs", label: "训练配置", icon: Layers, routeKey: "configs", group: "tools" },
  { to: "/datasets", label: "数据集", icon: Database, routeKey: "datasets", group: "tools" },
  { to: "/image-studio", label: "图像工作台", icon: Palette, routeKey: "image-studio", group: "tools" },
  { to: "/gallery", label: "样图画廊", icon: Images, routeKey: "gallery", group: "tools" },
  { to: "/settings", label: "设置", icon: Settings, routeKey: "settings", group: "system" },
  { to: "/about", label: "关于", icon: Info, routeKey: "about", group: "system" },
]

const NAV_GROUPS: Array<{ key: string; label: string }> = [
  { key: "workspace", label: "工作区" },
  { key: "tools", label: "工具" },
  { key: "system", label: "系统" },
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
    const prevent = (e: Event) => e.preventDefault()
    document.addEventListener("contextmenu", prevent)
    return () => {
      document.removeEventListener("contextmenu", prevent)
    }
  }, [])

  const prefetchRoute = useCallback((routeKey: AppRouteModuleKey) => {
    void preloadAppRoute(routeKey)
  }, [])

  const isRouteActive = (href: string) =>
    href === "/"
      ? location.pathname === "/"
      : location.pathname === href || location.pathname.startsWith(`${href}/`)

  const resolvedTitle =
    NAV.find((n) => isRouteActive(n.to))?.label ?? "LoraHub"

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
        <SidebarHeader className="border-b border-sidebar-border/70 px-4 py-4">
          <div className="px-1">
            <div className="flex items-center justify-between gap-2">
              <div className="shiro-kicker">LoraHub</div>
              <div className="shiro-microcopy">v{__APP_VERSION__}</div>
            </div>
            <div className="mt-1 text-xs leading-5 text-sidebar-foreground/66">
              LoRA 训练工作台
            </div>
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
                        render={<NavLink to={item.to} end={item.to === "/"} />}
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
          <div className="space-y-3 px-1">
            <div className="space-y-1.5">
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
          </div>
        </SidebarFooter>
      </Sidebar>

      {/* --- Main content --- */}
      <SidebarInset className="min-h-screen bg-transparent">
        <div className="relative flex min-h-screen flex-1 flex-col">
          {/* Header */}
          <header className="sticky top-0 z-30">
            <div className="shiro-toolbar px-3 py-3 md:px-4">
              <div className="flex items-center gap-3">
                <SidebarTrigger />
                <div className="hidden h-8 w-px bg-border/70 md:block" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-3">
                    <div className="shiro-kicker">LoraHub</div>
                    <div className="shiro-microcopy">[SYS.OK]</div>
                  </div>
                  <div className="mt-1 truncate text-base font-semibold tracking-[-0.01em]">
                    {resolvedTitle}
                  </div>
                </div>
              </div>
            </div>
            <GlobalStatusBar />
          </header>

          {/* Page content */}
          <main className="mx-auto flex w-full max-w-[1720px] min-w-0 flex-1 flex-col px-4 py-5 md:px-6 md:py-6 xl:px-8">
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
          </main>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
