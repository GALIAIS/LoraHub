import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Images, Search, SlidersHorizontal, X } from "lucide-react"
import { api, type SampleGalleryItem } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Pagination } from "@/components/ui/pagination"
import { cn } from "@/lib/utils"
import {
  JobFilterChips,
  SampleLightbox,
  SampleTile,
} from "./gallery-components"

const PAGE_SIZE_OPTIONS = [24, 48, 96, 192]

export function GalleryPage() {
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([])
  const [selectedConfigs, setSelectedConfigs] = useState<string[]>([])
  const [search, setSearch] = useState("")
  const [active, setActive] = useState<SampleGalleryItem | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(48)

  // Reset to page 1 whenever the filter changes — otherwise the user can
  // be parked on page 5 of an empty filter result.
  useEffect(() => {
    setPage(1)
  }, [selectedJobIds, selectedConfigs, search, pageSize])

  const jobs = useJobsList()

  // Pause the slow background refresh while a lightbox is open or the
  // tab is hidden — re-fetching the entire feed every 6s is wasted work
  // when the user can't see the grid.
  const [docHidden, setDocHidden] = useState(
    typeof document !== "undefined" ? document.hidden : false,
  )
  useEffect(() => {
    if (typeof document === "undefined") return
    const onChange = () => setDocHidden(document.hidden)
    document.addEventListener("visibilitychange", onChange)
    return () => document.removeEventListener("visibilitychange", onChange)
  }, [])
  const refetchInterval = active != null || docHidden ? false : 6000

  // Refresh the gallery on a slow cadence so freshly produced samples
  // appear without forcing a manual reload, but not so fast that every
  // mouse-move re-fetches the entire feed.
  const offset = (page - 1) * pageSize
  const samples = useQuery({
    queryKey: ["samples", selectedJobIds, pageSize, offset],
    queryFn: () =>
      api.listSamples({
        limit: pageSize,
        offset,
        jobIds: selectedJobIds.length > 0 ? selectedJobIds : undefined,
      }),
    refetchInterval,
  })

  const items = samples.data?.items ?? []
  const total = samples.data?.total ?? 0
  const allJobs = jobs.data?.jobs ?? []

  // Client-side narrowing on top of the server-side job filter. Search
  // hits filename and config name; config chips narrow to specific
  // configs. The server doesn't (yet) expose either as parameters, so
  // it's a "filter the visible page" — good enough since most users
  // pair this with a small selectedJobIds.
  const configOptions = useMemo(() => {
    const seen = new Map<string, number>()
    for (const it of items) {
      if (!it.config_name) continue
      seen.set(it.config_name, (seen.get(it.config_name) ?? 0) + 1)
    }
    return Array.from(seen.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([name, count]) => ({ name, count }))
  }, [items])

  const visibleItems = useMemo(() => {
    const q = search.trim().toLowerCase()
    return items.filter((item) => {
      if (selectedConfigs.length > 0) {
        if (!item.config_name || !selectedConfigs.includes(item.config_name)) {
          return false
        }
      }
      if (q) {
        const filename = item.path.split(/[\\/]/).pop() ?? ""
        const haystack = [filename, item.config_name ?? "", item.job_name].join(" ").toLowerCase()
        if (!haystack.includes(q)) return false
      }
      return true
    })
  }, [items, selectedConfigs, search])

  // Lightbox arrow-key navigation — mapped against visibleItems so
  // the user flips through whatever is currently shown after client
  // filters, not the full server response.
  const activeIndex = active
    ? visibleItems.findIndex(
        (it) => it.job_id === active.job_id && it.path === active.path,
      )
    : -1
  useEffect(() => {
    if (active == null) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft" && activeIndex > 0) {
        event.preventDefault()
        setActive(visibleItems[activeIndex - 1])
      } else if (
        event.key === "ArrowRight" &&
        activeIndex >= 0 &&
        activeIndex < visibleItems.length - 1
      ) {
        event.preventDefault()
        setActive(visibleItems[activeIndex + 1])
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [active, activeIndex, visibleItems])

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-4 py-4 md:px-6 md:py-5 space-y-4 w-full">
        <Card size="sm">
          <CardContent className="px-3 py-3 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative min-w-[14rem] flex-1">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/70" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="搜文件名 / 配置名 / 任务名"
                  className="h-8 pl-8 text-[12px]"
                />
              </div>
              <Pagination
                total={total}
                pageSize={pageSize}
                page={page}
                onPageChange={setPage}
                pageSizeOptions={PAGE_SIZE_OPTIONS}
                onPageSizeChange={setPageSize}
              />
              {(selectedJobIds.length > 0 ||
                selectedConfigs.length > 0 ||
                search.trim()) && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setSelectedJobIds([])
                    setSelectedConfigs([])
                    setSearch("")
                  }}
                  className="h-7 text-[11px]"
                >
                  <X className="size-3" /> 清除筛选
                </Button>
              )}
            </div>
            <details className="rounded-[4px] border border-border/60 bg-muted/20">
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-[11px] text-muted-foreground hover:text-foreground">
                <SlidersHorizontal className="size-3.5" />
                <span className="font-medium text-foreground">筛选条件</span>
                <span className="tabular-nums">
                  任务 {selectedJobIds.length || "全部"} / 配置 {selectedConfigs.length || "全部"}
                </span>
              </summary>
              <div className="max-h-[14rem] space-y-3 overflow-y-auto border-t border-border/60 px-3 py-3">
                <JobFilterChips
                  jobs={allJobs}
                  selectedJobIds={selectedJobIds}
                  onToggle={(id) =>
                    setSelectedJobIds((prev) =>
                      prev.includes(id)
                        ? prev.filter((p) => p !== id)
                        : [...prev, id],
                    )
                  }
                  onSelectAll={() =>
                    setSelectedJobIds(allJobs.map((j) => j.id))
                  }
                  onClear={() => setSelectedJobIds([])}
                />
                {configOptions.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-muted-foreground/70">
                      <span>按配置筛选</span>
                      {selectedConfigs.length > 0 && (
                        <button
                          type="button"
                          onClick={() => setSelectedConfigs([])}
                          className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground hover:text-foreground transition-colors"
                        >
                          清除
                        </button>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {configOptions.map(({ name, count }) => {
                        const isActive = selectedConfigs.includes(name)
                        return (
                          <button
                            key={name}
                            type="button"
                            onClick={() =>
                              setSelectedConfigs((prev) =>
                                prev.includes(name)
                                  ? prev.filter((p) => p !== name)
                                  : [...prev, name],
                              )
                            }
                            className={cn(
                              "rounded-[3px] border px-2 py-1 text-[11px] transition-colors",
                              isActive
                                ? "border-primary bg-primary/10 text-primary"
                                : "border-border/60 bg-background/60 text-muted-foreground hover:bg-muted/40",
                            )}
                          >
                            <span>{name}</span>
                            <span className="ml-1 text-muted-foreground/70 tabular-nums">
                              ({count})
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            </details>
          </CardContent>
        </Card>

        {samples.isLoading && (
          <div className="text-center text-sm text-muted-foreground py-12">
            正在加载样图…
          </div>
        )}

        {samples.isError && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs font-mono text-destructive">
            {(samples.error as Error).message}
          </div>
        )}

        {!samples.isLoading && items.length === 0 && (
          <div className="rounded-[6px] border border-dashed border-border/70 bg-muted/30 px-6 py-16 text-center">
            <Images className="size-8 mx-auto text-muted-foreground/60" />
            <div className="mt-3 text-sm text-muted-foreground">
              {selectedJobIds.length > 0
                ? "选中的任务还没有样图。"
                : "还没有任何任务产出样图。"}
            </div>
          </div>
        )}

        {!samples.isLoading && items.length > 0 && visibleItems.length === 0 && (
          <div className="rounded-[6px] border border-dashed border-border/70 bg-muted/30 px-6 py-12 text-center">
            <Images className="size-7 mx-auto text-muted-foreground/60" />
            <div className="mt-3 text-sm text-muted-foreground">
              当前页 {items.length} 张样图均未匹配筛选条件。
            </div>
            <Button
              variant="outline"
              size="sm"
              className="mt-3 h-7 text-[11px]"
              onClick={() => {
                setSelectedConfigs([])
                setSearch("")
              }}
            >
              清除客户端筛选
            </Button>
          </div>
        )}

        {visibleItems.length > 0 && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {visibleItems.map((item) => (
                <SampleTile
                  key={`${item.job_id}:${item.path}`}
                  item={item}
                  onOpen={() => setActive(item)}
                />
              ))}
            </div>
            <Pagination
              total={total}
              pageSize={pageSize}
              page={page}
              onPageChange={setPage}
              pageSizeOptions={PAGE_SIZE_OPTIONS}
              onPageSizeChange={setPageSize}
            />
          </>
        )}
      </div>

      <SampleLightbox
        item={active}
        onClose={() => setActive(null)}
        onPrev={() =>
          activeIndex > 0 ? setActive(visibleItems[activeIndex - 1]) : undefined
        }
        onNext={() =>
          activeIndex >= 0 && activeIndex < visibleItems.length - 1
            ? setActive(visibleItems[activeIndex + 1])
            : undefined
        }
        hasPrev={activeIndex > 0}
        hasNext={activeIndex >= 0 && activeIndex < visibleItems.length - 1}
      />
    </div>
  )
}
