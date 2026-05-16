import { useEffect, useState } from "react"
import { NavLink, Outlet } from "react-router-dom"
import {
  Activity,
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
import { GlobalStatusBar } from "@/components/global-status-bar"
import { cn } from "@/lib/utils"

const NAV = [
  { to: "/", label: "数据面板", icon: Activity },
  { to: "/jobs", label: "训练任务", icon: ListTree },
  { to: "/sweeps", label: "超参 sweep", icon: SlidersHorizontal },
  { to: "/configs", label: "训练配置", icon: Layers },
  { to: "/datasets", label: "数据集", icon: Database },
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

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      <aside className="w-56 shrink-0 border-r border-sidebar-border bg-sidebar/95 backdrop-blur px-3 py-5 flex flex-col gap-1 overflow-y-auto">
        <div className="px-2 mb-5 flex items-center gap-2">
          <div className="size-8 rounded-[6px] bg-primary text-primary-foreground grid place-items-center font-semibold tracking-tight text-sm">
            L
          </div>
          <div className="flex-1">
            <div className="text-[13px] font-semibold leading-tight">LoraHub</div>
            <div className="text-[11px] text-muted-foreground">训练工作台</div>
          </div>
        </div>

        <nav className="flex flex-col gap-0.5">
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
              <Icon className="size-3.5" />
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

          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
            v0.2
          </div>
        </div>
      </aside>

      <main className="flex-1 min-w-0 h-full overflow-hidden flex flex-col">
        <GlobalStatusBar />
        <div className="flex-1 min-h-0 overflow-hidden">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
