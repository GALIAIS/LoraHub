import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  PanelLeftOpen,
  SlidersHorizontal,
} from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { SweepDetailPanel } from "./sweep-detail"
import { SweepSidebar } from "./sweep-sidebar"

const SIDEBAR_KEY = "lorahub.sweeps.sidebar"

export function SweepsPage() {
  const sweeps = useQuery({
    queryKey: ["sweeps"],
    queryFn: api.listSweeps,
    refetchInterval: 4000,
    // refetchInterval 已经每 4s 拉一次,挂载/可见性变化时不需要再
    // 立刻补一发 — staleTime 顶住短时窗口的重复请求。
    staleTime: 2_000,
  })
  const list = sweeps.data?.sweeps ?? []

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    return window.localStorage.getItem(SIDEBAR_KEY) !== "closed"
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "open" : "closed")
  }, [sidebarOpen])

  // Auto-pick first sweep when nothing is selected and the list arrives;
  // also self-heal when the active sweep gets dropped from the registry.
  useEffect(() => {
    if (list.length === 0) {
      if (selectedId !== null) setSelectedId(null)
      return
    }
    const known = new Set(list.map((s) => s.sweep_id))
    if (selectedId === null || !known.has(selectedId)) {
      setSelectedId(list[0].sweep_id)
    }
  }, [list, selectedId])

  return (
    <div
      className={cn(
        "grid h-full min-h-0 overflow-hidden grid-rows-[1fr] transition-[grid-template-columns] duration-200",
        sidebarOpen
          ? "grid-cols-[minmax(240px,300px)_1fr]"
          : "grid-cols-[0px_1fr]",
      )}
    >
      <aside
        className={cn(
          "shiro-page-aside flex flex-col min-h-0 min-w-0 overflow-hidden",
          !sidebarOpen && "pointer-events-none opacity-0",
        )}
        aria-hidden={!sidebarOpen}
      >
        <SweepSidebar
          list={list}
          loading={sweeps.isLoading}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onClose={() => setSidebarOpen(false)}
        />
      </aside>

      <section className="min-w-0 min-h-0 flex flex-col bg-background/60 overflow-hidden relative">
        {!sidebarOpen && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSidebarOpen(true)}
            className="absolute left-3 top-3 z-10 shadow-[var(--panel-shadow)]"
            title="展开侧栏"
          >
            <PanelLeftOpen className="size-4" />
            <span className="ml-1 text-xs">{list.length} 个 sweep</span>
          </Button>
        )}
        {selectedId ? (
          <SweepDetailPanel sweepId={selectedId} />
        ) : (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="size-4" />
              选择左侧的 sweep 以查看详情。
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
