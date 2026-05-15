import { Outlet, NavLink } from "react-router-dom"
import { Activity, ListTree, Layers, Database, Settings } from "lucide-react"
import { cn } from "@/lib/utils"

const NAV = [
  { to: "/", label: "数据面板", icon: Activity },
  { to: "/jobs", label: "训练任务", icon: ListTree },
  { to: "/recipes", label: "训练配方", icon: Layers },
  { to: "/datasets", label: "数据集", icon: Database },
  { to: "/settings", label: "设置", icon: Settings },
]

export default function App() {
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
        <div className="mt-auto px-2 pt-4 text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
          v0.2
        </div>
      </aside>

      <main className="flex-1 min-w-0 h-full overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
