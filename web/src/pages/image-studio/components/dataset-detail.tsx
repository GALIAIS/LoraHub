import { useCallback, useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import {
  imageStudioList,
  imageStudioGetImage,
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
  imageStudioBatchDelete,
} from "@/lib/api"
import type { ImageStudioItem } from "@/lib/api"
import { removeTask } from "@/lib/studio-task-store"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { FilterPanel } from "./filter-panel"
import { ImageGrid } from "./image-grid"
import { Inspector } from "./inspector"
import { LightboxModal } from "./lightbox-modal"
import { PendingOpsDialog } from "./pending-ops-dialog"
import { UploadDropZone } from "./upload-zone"
import { BatchToolbar } from "./batch-toolbar"
import { HelpOverlay } from "./help-overlay"
import { AiBulkModal } from "./ai-bulk-modal"
import { TaggingPanel } from "./tagging-panel"
import { DuplicatesView } from "./duplicates-view"
import { type FilterState, defaultFilters, applyFilters } from "./types"
import type { AiBulkTab } from "./types"
import { Pagination, StudioTaskBanner } from "./dataset-detail-widgets"
import { startDatasetAiBulkTask } from "./dataset-detail-ai"
import {
  useDatasetPendingOpsCount,
  useDatasetTaskState,
} from "./dataset-detail-hooks"
import { DatasetDetailToolbar } from "./dataset-detail-toolbar"
import { TriggerWordSummary } from "./trigger-word-summary"

export function DatasetDetail() {
  const [params, setParams] = useSearchParams()
  const path = params.get("path") || ""
  const pageRaw = Number.parseInt(params.get("page") || "1", 10)
  const page = Number.isFinite(pageRaw) && pageRaw >= 1 ? pageRaw : 1
  const sort = params.get("sort") || "name"
  const recursive = params.get("recursive") === "1"
  const view = params.get("view") || "grid"

  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [multiSelected, setMultiSelected] = useState<Set<string>>(new Set())
  const [showHelp, setShowHelp] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [showAiBulk, setShowAiBulk] = useState(false)
  const [showTagging, setShowTagging] = useState(false)
  const [showOpsQueue, setShowOpsQueue] = useState(false)
  // Lightbox is path-based so the modal can flip through whatever the
  // current filtered grid is showing — we resolve to an index lazily.
  const [lightboxPath, setLightboxPath] = useState<string | null>(null)
  const [filters, setFilters] = useState<FilterState>(defaultFilters)
  // AlertDialog targets — pinning either of these opens the modal;
  // confirming triggers the destructive action, cancelling clears.
  const [pendingDeleteSingle, setPendingDeleteSingle] =
    useState<ImageStudioItem | null>(null)
  const [pendingDeleteBulk, setPendingDeleteBulk] = useState(false)
  const queryClient = useQueryClient()
  const datasetName = path.split(/[/\\]/).pop() || ""

  const pendingOpsCount = useDatasetPendingOpsCount(path)
  const { activeStudioTask, triggerWordTop, setTriggerWordTop } =
    useDatasetTaskState(path)

  const setPage = useCallback(
    (p: number) => {
      const next = new URLSearchParams(params)
      next.set("page", String(Math.max(1, p)))
      setSelectedPath(null)
      setMultiSelected(new Set())
      setParams(next)
    },
    [params, setParams],
  )

  const goBack = () => setParams(new URLSearchParams())

  const listQuery = useQuery({
    queryKey: ["image-studio", "list", path, page, sort, recursive],
    queryFn: () => imageStudioList({ path, page, limit: 48, sort, recursive }),
    enabled: !!path,
  })

  const detailQuery = useQuery({
    queryKey: ["image-studio", "detail", selectedPath],
    queryFn: () => imageStudioGetImage(selectedPath!),
    enabled: !!selectedPath,
  })

  const batchDeleteMutation = useMutation({
    mutationFn: () => imageStudioBatchDelete({ paths: Array.from(multiSelected) }),
    onSuccess: () => {
      setMultiSelected(new Set())
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    },
  })

  const batchFavMutation = useMutation({
    mutationFn: async () => {
      // Fan out the per-image saves in parallel — previously this was
      // a sequential `for...of await` loop, which made marking 1000
      // images take ~1000× longer than necessary. We still surface
      // the first error encountered (Promise.all rejects on the first
      // failure) so the toast caller can show a useful message.
      await Promise.all(
        Array.from(multiSelected).map((p) =>
          imageStudioSaveAnnotation({ path: p, favorite: true }),
        ),
      )
    },
    onSuccess: () => {
      setMultiSelected(new Set())
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    },
  })

  // Context menu actions for image tiles
  const handleContextAction = useCallback(
    (action: string, item: ImageStudioItem) => {
      switch (action) {
        case "inspect":
          setSelectedPath(item.path)
          break
        case "edit-caption":
          setSelectedPath(item.path)
          break
        case "toggle-fav":
          imageStudioSaveAnnotation({
            path: item.path,
            favorite: !item.annotation?.favorite,
          }).then(() => queryClient.invalidateQueries({ queryKey: ["image-studio"] }))
          break
        case "rotate":
          imageStudioAddOp({ path: item.path, op: "rotate", payload: { degrees: 90 } })
            .then(() => imageStudioApplyOps(item.path))
            .then(() => queryClient.invalidateQueries({ queryKey: ["image-studio"] }))
          break
        case "flip":
          imageStudioAddOp({ path: item.path, op: "flip", payload: { direction: "horizontal" } })
            .then(() => imageStudioApplyOps(item.path))
            .then(() => queryClient.invalidateQueries({ queryKey: ["image-studio"] }))
          break
        case "copy-path":
          navigator.clipboard.writeText(item.path)
          break
        case "lightbox":
          setLightboxPath(item.path)
          break
        case "delete":
          setPendingDeleteSingle(item)
          break
      }
    },
    [queryClient],
  )

  // Multi-select toggle
  const toggleMultiSelect = useCallback((itemPath: string) => {
    setMultiSelected((prev) => {
      const next = new Set(prev)
      if (next.has(itemPath)) next.delete(itemPath)
      else next.add(itemPath)
      return next
    })
  }, [])

  // Enhanced keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const items = listQuery.data?.items
      if (!items) return
      const currentIdx = selectedPath
        ? items.findIndex((i) => i.path === selectedPath)
        : -1

      switch (e.key) {
        case "j":
          e.preventDefault()
          if (currentIdx < items.length - 1) setSelectedPath(items[currentIdx + 1].path)
          else if (currentIdx === -1 && items.length > 0) setSelectedPath(items[0].path)
          break
        case "k":
          e.preventDefault()
          if (currentIdx > 0) setSelectedPath(items[currentIdx - 1].path)
          break
        case " ": // space = preview
          e.preventDefault()
          if (currentIdx >= 0) setSelectedPath(items[currentIdx].path)
          break
        case "x": // toggle multi-select
          e.preventDefault()
          if (currentIdx >= 0) toggleMultiSelect(items[currentIdx].path)
          break
        case "e": // edit caption
          e.preventDefault()
          if (currentIdx >= 0) setSelectedPath(items[currentIdx].path)
          break
        case "f": // open the focused image full-screen
          e.preventDefault()
          if (currentIdx >= 0) setLightboxPath(items[currentIdx].path)
          break
        case "d": // soft-delete current focused tile
          e.preventDefault()
          if (currentIdx >= 0) {
            // Open the existing single-delete confirm dialog instead
            // of nuking immediately — keyboard-trigger destructives
            // without confirm is too easy to fire by accident.
            setPendingDeleteSingle(items[currentIdx])
          }
          break
        case "a":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            setMultiSelected(new Set(items.map((i) => i.path)))
          }
          break
        case "Escape":
          if (multiSelected.size > 0) setMultiSelected(new Set())
          else {
            setSelectedPath(null)
            setShowHelp(false)
          }
          break
        case "?":
          e.preventDefault()
          setShowHelp((v) => !v)
          break
      }
    },
    [listQuery.data?.items, selectedPath, multiSelected, queryClient, toggleMultiSelect],
  )

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])

  const data = listQuery.data
  const totalPages = data ? Math.ceil(data.total / data.limit) : 0
  const displayPage = data?.page ?? page
  const filteredItems = data ? applyFilters(data.items, filters) : []

  const handleAiBulkStart = async (tab: AiBulkTab, params: Record<string, unknown>) => {
    setShowAiBulk(false)

    try {
      await startDatasetAiBulkTask({ tab, params, path, recursive })
    } catch (err: unknown) {
      // Top-level catch is for failures BEFORE a task record made it into
      // the store (e.g. /smart-caption submit returned 4xx). Surface via
      // toast since there's no banner to attach the error to.
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("AI 批量任务启动失败", { description: msg })
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <DatasetDetailToolbar
        datasetName={datasetName}
        path={path}
        total={data?.total}
        sort={sort}
        view={view}
        recursive={recursive}
        params={params}
        setParams={setParams}
        showFilters={showFilters}
        showTagging={showTagging}
        pendingOpsCount={pendingOpsCount}
        onBack={goBack}
        onToggleFilters={() => setShowFilters((v) => !v)}
        onToggleTagging={() => setShowTagging((v) => !v)}
        onOpenAiBulk={() => setShowAiBulk(true)}
        onOpenOpsQueue={() => setShowOpsQueue(true)}
        onOpenHelp={() => setShowHelp(true)}
      />

      {/* Upload drop zone */}
      <UploadDropZone
        datasetName={datasetName}
        onComplete={() => queryClient.invalidateQueries({ queryKey: ["image-studio"] })}
      />

      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}
      {showAiBulk && (
        <AiBulkModal
          paths={Array.from(multiSelected)}
          datasetPath={path}
          onClose={() => setShowAiBulk(false)}
          onStart={handleAiBulkStart}
        />
      )}

      {/* AI progress bar */}
      {activeStudioTask && (
        <StudioTaskBanner
          task={activeStudioTask}
          onDismiss={() => removeTask(activeStudioTask.id)}
        />
      )}

      {triggerWordTop && triggerWordTop.length > 0 && (
        <TriggerWordSummary
          candidates={triggerWordTop}
          onClose={() => setTriggerWordTop(null)}
        />
      )}

      {/* Main content */}
      {view === "duplicates" ? (
        <DuplicatesView path={path} recursive={recursive} />
      ) : (
        <div className="flex min-h-0 flex-1 overflow-hidden">
          {showFilters && (
            <FilterPanel
              filters={filters}
              onChange={setFilters}
              onClose={() => setShowFilters(false)}
              items={data?.items ?? []}
              onReset={() => setFilters(defaultFilters)}
            />
          )}

          <div className="min-w-0 flex-1 overflow-y-auto p-3">
            {listQuery.isLoading && (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                加载中…
              </div>
            )}
            {data && filteredItems.length === 0 && !listQuery.isLoading && (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                {data.items.length === 0
                  ? "该目录下未找到图片，拖入文件或压缩包上传"
                  : "当前筛选条件下无匹配图片"}
              </div>
            )}
            {filteredItems.length > 0 && (
              <>
                <ImageGrid
                  items={filteredItems}
                  selectedPath={selectedPath}
                  multiSelected={multiSelected}
                  onSelect={setSelectedPath}
                  onMultiToggle={toggleMultiSelect}
                  onDoubleSelect={(p) => setLightboxPath(p)}
                  onContextAction={handleContextAction}
                />
                {totalPages > 1 && (
                  <Pagination page={displayPage} total={totalPages} onChange={setPage} />
                )}
              </>
            )}
          </div>

          {showTagging && (
            <aside className="shiro-page-aside w-56 shrink-0 overflow-y-auto">
              <TaggingPanel datasetPath={path} />
            </aside>
          )}

          {selectedPath && (
            <Inspector
              detail={detailQuery.data ?? null}
              loading={detailQuery.isLoading}
              path={selectedPath}
              onClose={() => setSelectedPath(null)}
              onOpenLightbox={() => setLightboxPath(selectedPath)}
            />
          )}
        </div>
      )}

      {/* Batch toolbar */}
      <BatchToolbar
        count={multiSelected.size}
        onDelete={() => setPendingDeleteBulk(true)}
        onFavorite={() => batchFavMutation.mutate()}
        onAiBulk={() => setShowAiBulk(true)}
        onExport={() => {
          // Minimum-viable export: copy a newline-separated list of
          // selected paths to the clipboard. Users can pipe that into
          // any zip / rsync / cp -t workflow they already have. A
          // proper server-side zip endpoint would be a separate
          // milestone (see audit B5).
          const paths = Array.from(multiSelected)
          if (paths.length === 0) return
          navigator.clipboard
            .writeText(paths.join("\n"))
            .then(() =>
              toast.success(`已复制 ${paths.length} 条路径`, {
                description: "粘贴到任意终端 / 资源管理器即可批量处理",
              }),
            )
            .catch((err) =>
              toast.error("复制失败", {
                description: err instanceof Error ? err.message : String(err),
              }),
            )
        }}
        onClear={() => setMultiSelected(new Set())}
      />

      <AlertDialog
        open={!!pendingDeleteSingle}
        onOpenChange={(open) => !open && setPendingDeleteSingle(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除图片</AlertDialogTitle>
            <AlertDialogDescription>
              将永久移除{" "}
              <code className="font-mono">{pendingDeleteSingle?.name}</code>
              {" "}及其标注。此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                const target = pendingDeleteSingle
                if (!target) return
                imageStudioAddOp({
                  path: target.path,
                  op: "delete",
                  payload: {},
                })
                  .then(() => imageStudioApplyOps(target.path))
                  .then(() =>
                    queryClient.invalidateQueries({
                      queryKey: ["image-studio"],
                    }),
                  )
                  .then(() => toast.success("已删除"))
                  .catch((err) =>
                    toast.error("删除失败", {
                      description:
                        err instanceof Error ? err.message : String(err),
                    }),
                  )
                setPendingDeleteSingle(null)
              }}
            >
              确认删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={pendingDeleteBulk}
        onOpenChange={(open) => !open && setPendingDeleteBulk(false)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>批量删除图片</AlertDialogTitle>
            <AlertDialogDescription>
              将删除选中的 {multiSelected.size} 张图片及其标注，移动到
              <code className="font-mono mx-1">_image_studio_trash/</code>
              ，可在文件管理器中找回。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                batchDeleteMutation.mutate()
                setPendingDeleteBulk(false)
              }}
            >
              确认删除 ({multiSelected.size})
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <LightboxModal
        open={lightboxPath != null}
        items={filteredItems}
        index={
          lightboxPath != null
            ? Math.max(
                0,
                filteredItems.findIndex((it) => it.path === lightboxPath),
              )
            : 0
        }
        onIndexChange={(idx) => {
          const next = filteredItems[idx]
          if (next) setLightboxPath(next.path)
        }}
        onClose={() => setLightboxPath(null)}
        onToggleFavorite={(item) => {
          imageStudioSaveAnnotation({
            path: item.path,
            favorite: !item.annotation?.favorite,
          }).then(() =>
            queryClient.invalidateQueries({ queryKey: ["image-studio"] }),
          )
        }}
      />

      <PendingOpsDialog
        open={showOpsQueue}
        onOpenChange={setShowOpsQueue}
        path={path}
      />
    </div>
  )
}
