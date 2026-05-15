import { Outlet, NavLink } from "react-router-dom"
import { Activity, ListTree, Layers, Database, Settings } from "lucide-react"
import { cn } from "@/lib/utils"

const NAV = [
  { to: "/", label: "Dashboard", icon: Activity },
  { to: "/jobs", label: "Jobs", icon: ListTree },
  { to: "/recipes", label: "Recipes", icon: Layers },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/settings", label: "Settings", icon: Settings },
]

export default function App() {
  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <aside className="w-56 border-r border-sidebar-border bg-sidebar/95 backdrop-blur px-3 py-5 flex flex-col gap-1">
        <div className="px-2 mb-5 flex items-center gap-2">
          <div className="size-8 rounded-[6px] bg-primary text-primary-foreground grid place-items-center font-semibold tracking-tight text-sm">
            L
          </div>
          <div className="flex-1">
            <div className="text-[13px] font-semibold leading-tight">LoraHub</div>
            <div className="text-[11px] text-muted-foreground">Training Workbench</div>
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

      <main className="flex-1 min-w-0">
        <Outlet />
      </main>
    </div>
  )
}
