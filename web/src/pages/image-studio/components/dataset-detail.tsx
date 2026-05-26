import { useCallback, useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  Filter,
  FolderOpen,
  HelpCircle,
  ListChecks,
  Loader2,
  Sparkles,
  Tag,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioList,
  imageStudioGetImage,
  imageStudioListOps,
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
  imageStudioBatchDelete,
  imageStudioBatchQuality,
  imageStudioBatchTriggerWords,
  startSmartCaptionSession,
  startTaggingSession,
} from "@/lib/api"
import type { ImageStudioItem } from "@/lib/api"
import {
  addTask,
  removeTask,
  updateTask,
  type StudioTaskRecord,
} from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"
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
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
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

export function DatasetDetail() {
  const [params, setParams] = useSearchParams()
  const path = params.get("path") || ""
  const page = Number(params.get("page") || "1")
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
  // Dataset-level trigger word ranking surfaced after a trigger-words
  // batch completes. Stays around so the user can copy a candidate into
  // the smart-caption "trigger word" field on the next run without
  // having to re-run analysis.
  const [triggerWordTop, setTriggerWordTop] = useState<
    { trigger: string; count: number }[] | null
  >(null)

  const queryClient = useQueryClient()
  const datasetName = path.split(/[/\\]/).pop() || ""

  // Lightweight pending-ops counter so the toolbar can flag the badge
  // when the user has unflushed edits. Refetches every 5s and on
  // cache invalidation (image-studio mutations bust everything under
  // the "image-studio" key, so this stays in sync without bookkeeping).
  const opsCountQuery = useQuery({
    queryKey: ["image-studio", "ops-count", path],
    queryFn: () => imageStudioListOps(path),
    enabled: !!path,
    refetchInterval: 5000,
    select: (data) => data.ops.length,
  })
  const pendingOpsCount = opsCountQuery.data ?? 0

  // Subscribe to the global studio task store so the AI progress banner
  // is driven by an out-of-tree polling loop instead of a per-component
  // setInterval. This makes the banner survive route changes (the page
  // unmounts on navigation, but the server keeps running and the store
  // keeps polling) and even hard reloads for sessioned task kinds.
  const studioTasks = useStudioTasksFor(path)
  // Newest first so the banner reflects whatever the user just kicked
  // off — multiple concurrent tasks are rare but tolerable.
  const activeStudioTask: StudioTaskRecord | null =
    studioTasks.length === 0
      ? null
      : [...studioTasks].sort((a, b) => b.startedAt - a.startedAt)[0]

  // When a sessioned task transitions to a terminal state, refresh the
  // dataset listing so any new annotations / scores light up. Keying
  // the effect on a derived signature stops it from firing every poll.
  const terminalSig = studioTasks
    .filter((t) => t.status !== "running")
    .map((t) => `${t.id}:${t.status}`)
    .join(",")
  useEffect(() => {
    if (terminalSig) {
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    }
  }, [terminalSig, queryClient])

  const setPage = (p: number) => {
    const next = new URLSearchParams(params)
    next.set("page", String(p))
    setParams(next)
  }

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
  const filteredItems = data ? applyFilters(data.items, filters) : []

  const handleAiBulkStart = async (tab: AiBulkTab, params: Record<string, unknown>) => {
    setShowAiBulk(false)
    const taskPath = (params.path as string) || path

    // Synthetic id for kinds without a server session_id. crypto.randomUUID
    // is everywhere we run; the fallback is just defensive.
    const newId = (): string =>
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `studio-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

    try {
      switch (tab) {
        case "smart-caption": {
          const captionSource = (params.captionSource as "vlm" | "tags" | undefined) ?? "vlm"
          const submit = await startSmartCaptionSession({
            path: taskPath,
            recursive,
            device: params.device as string,
            mergeStrategy: params.mergeStrategy as string,
            captionMode: params.captionMode as "general" | "style" | "character",
            captionSource,
            triggerWord: params.triggerWord as string | undefined,
            stripStyleTags: params.stripStyleTags as boolean | undefined,
            skipExisting: params.skipExisting as boolean | undefined,
          })
          addTask({
            id: submit.session_id,
            kind: "smart-caption",
            datasetPath: taskPath,
            label: captionSource === "tags"
              ? "智能标注（WD14 + LLM 文本模式）"
              : "智能标注（WD14 + VLM 视觉模式）",
            total: submit.total,
          })
          // Polling is now driven by the global studio task store; nothing
          // else to do here beyond dropping the record into the store.
          return
        }
        case "wd14": {
          const session = await startTaggingSession({
            path: taskPath,
            tagger: (params.model_id as string)?.startsWith("joy") ? "joytag" : "wd14",
            model_id: params.model_id as string,
            general: params.general as number,
            character: params.character as number,
            device: params.device as string,
            overwrite: params.overwrite as boolean,
            recursive,
          })
          addTask({
            id: session.session_id,
            kind: "wd14",
            datasetPath: taskPath,
            label: "WD14 标注",
          })
          return
        }
        case "quality-score": {
          // Synchronous server endpoint with no session_id. The task is
          // tracked in the store as "in-flight" so the banner survives a
          // navigate-away within the same page-load. A hard reload mid-run
          // can't reconnect (no handle to look up) — that's documented
          // upstream and acceptable for these short batches.
          const id = newId()
          addTask({
            id,
            kind: "quality-score",
            datasetPath: taskPath,
            label: "质量评分",
            sessioned: false,
          })
          try {
            const res = await imageStudioBatchQuality({
              path: taskPath,
              recursive,
              skipScored: params.skipScored as boolean | undefined,
            })
            updateTask(id, {
              status: "completed",
              processed: res.processed,
              label: res.skipped
                ? `质量评分完成（跳过 ${res.skipped} 已评分）`
                : "质量评分完成",
            })
          } catch (err) {
            updateTask(id, {
              status: "failed",
              errorMsg: err instanceof Error ? err.message : String(err),
              label: "质量评分失败",
            })
          }
          break
        }
        case "trigger-words": {
          // Dedicated per-image trigger-word VLM analysis. Each image
          // gets 1-3 candidate phrases stored on its annotation, and
          // the response carries a dataset-level top-N ranking we
          // surface once the batch finishes.
          const id = newId()
          addTask({
            id,
            kind: "trigger-words",
            datasetPath: taskPath,
            label: "分析触发词",
            sessioned: false,
          })
          try {
            const res = await imageStudioBatchTriggerWords({
              path: taskPath,
              recursive,
              skipAnalyzed: params.skipAnalyzed as boolean | undefined,
            })
            setTriggerWordTop(res.dataset_top)
            updateTask(id, {
              status: "completed",
              processed: res.processed,
              label: res.skipped
                ? `触发词分析完成（跳过 ${res.skipped} 已分析）`
                : "触发词分析完成",
            })
          } catch (err) {
            updateTask(id, {
              status: "failed",
              errorMsg: err instanceof Error ? err.message : String(err),
              label: "触发词分析失败",
            })
          }
          break
        }
      }
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    } catch (err: unknown) {
      // Top-level catch is for failures BEFORE a task record made it into
      // the store (e.g. /smart-caption submit returned 4xx). Surface via
      // toast since there's no banner to attach the error to.
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("AI 批量任务启动失败", { description: msg })
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-border/60 px-4 py-2">
        <Button
          variant="ghost"
          size="sm"
          className="size-7 p-0"
          onClick={goBack}
          aria-label="返回数据集列表"
          title="返回数据集列表"
        >
          <FolderOpen className="size-4" />
        </Button>
        <span className="font-medium text-sm">{datasetName}</span>
        <span className="font-mono text-xs text-muted-foreground truncate flex-1">{path}</span>
        {data && (
          <span className="text-xs text-muted-foreground">{data.total} 张图片</span>
        )}
        <SortSelect
          value={sort}
          onChange={(s) => {
            const next = new URLSearchParams(params)
            next.set("sort", s)
            next.set("page", "1")
            setParams(next)
          }}
        />
        {/* View switch — same chip style as analysis page filter chips
            so the language stays consistent across the workbench. */}
        <div
          role="group"
          aria-label="视图模式"
          className="inline-flex h-7 items-center rounded-[4px] border border-border/60 bg-background overflow-hidden text-[11px]"
        >
          <ViewChip
            active={view === "grid"}
            onClick={() => {
              const n = new URLSearchParams(params)
              n.set("view", "grid")
              setParams(n)
            }}
          >
            网格
          </ViewChip>
          <span className="h-full w-px bg-border/60" aria-hidden />
          <ViewChip
            active={view === "duplicates"}
            onClick={() => {
              const n = new URLSearchParams(params)
              n.set("view", "duplicates")
              setParams(n)
            }}
          >
            去重
          </ViewChip>
        </div>
        <label className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer select-none">
          <Switch
            size="sm"
            checked={recursive}
            onCheckedChange={(checked) => {
              const n = new URLSearchParams(params)
              if (checked) n.set("recursive", "1")
              else n.delete("recursive")
              n.set("page", "1")
              setParams(n)
            }}
          />
          递归
        </label>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowFilters((v) => !v)}
          className={cn(
            "size-7 p-0",
            showFilters && "bg-muted text-primary",
          )}
          aria-label="筛选面板"
          aria-pressed={showFilters}
          title="筛选面板"
        >
          <Filter className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowTagging((v) => !v)}
          className={cn(
            "size-7 p-0",
            showTagging && "bg-muted text-primary",
          )}
          aria-label="WD14 标注"
          aria-pressed={showTagging}
          title="WD14 标注"
        >
          <Tag className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowAiBulk(true)}
          className="size-7 p-0"
          aria-label="AI 批量操作"
          title="AI 批量操作"
        >
          <Sparkles className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowOpsQueue(true)}
          className={cn(
            "relative size-7 p-0",
            pendingOpsCount > 0 && "text-primary",
          )}
          aria-label={`待应用操作 (${pendingOpsCount})`}
          title={
            pendingOpsCount > 0
              ? `${pendingOpsCount} 个待应用操作`
              : "待应用操作 (空)"
          }
        >
          <ListChecks className="size-4" />
          {pendingOpsCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-semibold leading-none text-primary-foreground">
              {pendingOpsCount > 99 ? "99+" : pendingOpsCount}
            </span>
          )}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowHelp(true)}
          className="size-7 p-0"
          aria-label="键盘快捷键 (?)"
          title="键盘快捷键 (?)"
        >
          <HelpCircle className="size-4" />
        </Button>
      </div>

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

      {/* Trigger word top-N (dataset-level summary, shown after a
          trigger-words batch finishes). Click a chip to copy it to
          the clipboard so the user can paste it into the smart-caption
          "trigger word" input on the next run. */}
      {triggerWordTop && triggerWordTop.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2 bg-muted/20">
          <span className="text-xs text-muted-foreground">数据集触发词候选（点击复制）：</span>
          {triggerWordTop.map((t) => (
            <button
              key={t.trigger}
              type="button"
              onClick={() => {
                void navigator.clipboard.writeText(t.trigger)
              }}
              className="inline-flex items-center gap-1 rounded border bg-background px-2 py-0.5 text-[11px] font-mono hover:bg-muted transition-colors"
              title={`${t.count} 张图片含此触发词 · 点击复制`}
            >
              <span>{t.trigger}</span>
              <span className="text-muted-foreground">·{t.count}</span>
            </button>
          ))}
          <button
            type="button"
            onClick={() => setTriggerWordTop(null)}
            className="ml-auto text-xs text-muted-foreground hover:text-foreground"
          >
            关闭
          </button>
        </div>
      )}

      {/* Main content */}
      {view === "duplicates" ? (
        <DuplicatesView path={path} recursive={recursive} />
      ) : (
        <div className="flex flex-1 min-h-0">
          {showFilters && (
            <FilterPanel
              filters={filters}
              onChange={setFilters}
              onClose={() => setShowFilters(false)}
              items={data?.items ?? []}
              onReset={() => setFilters(defaultFilters)}
            />
          )}

          <div className="flex-1 overflow-y-auto p-3">
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
                  <Pagination page={page} total={totalPages} onChange={setPage} />
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

function ViewChip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "px-2.5 h-full transition-colors",
        active
          ? "bg-muted font-medium text-foreground"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}

const SORT_OPTIONS = [
  { value: "name", label: "名称" },
  { value: "mtime", label: "修改时间" },
  { value: "size", label: "大小" },
] as const

function SortSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <Select
      items={SORT_OPTIONS}
      value={value}
      onValueChange={(v) => onChange(v as string)}
    >
      <SelectTrigger className="h-7 text-[11px] min-w-[6.5rem]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {SORT_OPTIONS.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function Pagination({
  page,
  total,
  onChange,
}: {
  page: number
  total: number
  onChange: (p: number) => void
}) {
  return (
    <div className="flex items-center justify-center gap-2 pt-4">
      <button
        type="button"
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
        className="rounded border px-2 py-1 text-xs disabled:opacity-40"
      >
        上一页
      </button>
      <span className="text-xs text-muted-foreground">
        {page} / {total}
      </span>
      <button
        type="button"
        disabled={page >= total}
        onClick={() => onChange(page + 1)}
        className="rounded border px-2 py-1 text-xs disabled:opacity-40"
      >
        下一页
      </button>
    </div>
  )
}

/**
 * Banner that summarises a single global studio task. The data shape
 * comes from the task store so the banner is identical no matter
 * whether the user just kicked the task off or just got back from
 * another route while it was running in the background.
 */
function StudioTaskBanner({
  task,
  onDismiss,
}: {
  task: StudioTaskRecord
  onDismiss: () => void
}) {
  const running = task.status === "running"
  const lastImageName = task.lastImage
    ? task.lastImage.split(/[/\\]/).pop() ?? ""
    : ""
  // Compose a label that mirrors what the legacy inline progress used to
  // show. For sessioned kinds we tack on the most recent image filename
  // so the user can tell the task is making forward progress.
  const label = running
    ? task.kind === "smart-caption"
      ? `${task.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
      : task.kind === "wd14"
        ? `${task.label}中… ${task.processed ?? 0}/${task.total ?? "?"}`
        : `${task.label}中…`
    : task.label

  return (
    <div className="flex items-center gap-3 border-b px-4 py-2 bg-muted/30">
      {running && <Loader2 className="size-4 animate-spin text-primary" />}
      <span className="text-xs font-medium">{label}</span>
      {task.processed != null && task.kind !== "wd14" && (
        <span className="text-xs text-muted-foreground">
          {task.processed}
          {task.total ? ` / ${task.total}` : ""} 张
        </span>
      )}
      {task.errorMsg && (
        <span className="text-xs text-destructive truncate flex-1">
          {task.errorMsg}
        </span>
      )}
      {!running && (
        <button
          type="button"
          onClick={onDismiss}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground"
        >
          关闭
        </button>
      )}
    </div>
  )
}
