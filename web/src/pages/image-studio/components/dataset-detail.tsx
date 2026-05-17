import { useCallback, useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Filter, FolderOpen, HelpCircle, Sparkles, Tag } from "lucide-react"
import {
  imageStudioList,
  imageStudioGetImage,
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
  imageStudioBatchDelete,
} from "@/lib/api"
import type { ImageStudioItem } from "@/lib/api"
import { FilterPanel } from "./filter-panel"
import { ImageGrid } from "./image-grid"
import { Inspector } from "./inspector"
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
  const [filters, setFilters] = useState<FilterState>(defaultFilters)

  const queryClient = useQueryClient()
  const datasetName = path.split(/[/\\]/).pop() || ""

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
      for (const p of multiSelected) {
        await imageStudioSaveAnnotation({ path: p, favorite: true })
      }
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
        case "delete":
          if (confirm(`确定要删除 "${item.name}" 吗？`)) {
            imageStudioAddOp({ path: item.path, op: "delete", payload: {} })
              .then(() => imageStudioApplyOps(item.path))
              .then(() => queryClient.invalidateQueries({ queryKey: ["image-studio"] }))
          }
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
        case "q": // quality vote
          e.preventDefault()
          break
        case "d": // soft-delete
          e.preventDefault()
          if (currentIdx >= 0) {
            imageStudioSaveAnnotation({ path: items[currentIdx].path, softDeleted: true })
              .then(() => queryClient.invalidateQueries({ queryKey: ["image-studio"] }))
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

  const handleAiBulkStart = (_tab: AiBulkTab, _params: Record<string, unknown>) => {
    // TODO: wire to actual API calls
    setShowAiBulk(false)
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b px-4 py-2">
        <button type="button" onClick={goBack} className="rounded p-1 hover:bg-muted" title="返回数据集列表">
          <FolderOpen className="size-4" />
        </button>
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
        <div className="flex rounded border text-xs">
          <button
            type="button"
            onClick={() => { const n = new URLSearchParams(params); n.set("view", "grid"); setParams(n) }}
            className={`px-2 py-1 ${view === "grid" ? "bg-muted font-medium" : ""}`}
          >
            网格
          </button>
          <button
            type="button"
            onClick={() => { const n = new URLSearchParams(params); n.set("view", "duplicates"); setParams(n) }}
            className={`px-2 py-1 ${view === "duplicates" ? "bg-muted font-medium" : ""}`}
          >
            去重
          </button>
        </div>
        <label className="flex items-center gap-1.5 text-xs cursor-pointer">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => {
              const n = new URLSearchParams(params)
              if (e.target.checked) n.set("recursive", "1")
              else n.delete("recursive")
              n.set("page", "1")
              setParams(n)
            }}
            className="size-3"
          />
          递归
        </label>
        <button
          type="button"
          onClick={() => setShowFilters((v) => !v)}
          className={`rounded p-1 hover:bg-muted ${showFilters ? "bg-muted text-primary" : "text-muted-foreground"}`}
          title="筛选面板"
        >
          <Filter className="size-4" />
        </button>
        <button
          type="button"
          onClick={() => setShowTagging((v) => !v)}
          className={`rounded p-1 hover:bg-muted ${showTagging ? "bg-muted text-primary" : "text-muted-foreground"}`}
          title="WD14 标注"
        >
          <Tag className="size-4" />
        </button>
        <button
          type="button"
          onClick={() => setShowAiBulk(true)}
          className="rounded p-1 text-muted-foreground hover:bg-muted"
          title="AI 批量操作"
        >
          <Sparkles className="size-4" />
        </button>
        <button
          type="button"
          onClick={() => setShowHelp(true)}
          className="rounded p-1 text-muted-foreground hover:bg-muted"
          title="键盘快捷键 (?)"
        >
          <HelpCircle className="size-4" />
        </button>
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
            />
          )}

          <div className="flex-1 overflow-y-auto p-3">
            {listQuery.isLoading && (
              <div className="flex items-center justify-center h-32 text-muted-foreground">
                加载中...
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
                  onContextAction={handleContextAction}
                />
                {totalPages > 1 && (
                  <Pagination page={page} total={totalPages} onChange={setPage} />
                )}
              </>
            )}
          </div>

          {showTagging && <TaggingPanel datasetPath={path} />}

          {selectedPath && (
            <Inspector
              detail={detailQuery.data ?? null}
              loading={detailQuery.isLoading}
              path={selectedPath}
              onClose={() => setSelectedPath(null)}
            />
          )}
        </div>
      )}

      {/* Batch toolbar */}
      <BatchToolbar
        count={multiSelected.size}
        onDelete={() => {
          if (confirm(`确定要删除选中的 ${multiSelected.size} 张图片吗？`)) {
            batchDeleteMutation.mutate()
          }
        }}
        onFavorite={() => batchFavMutation.mutate()}
        onAiBulk={() => setShowAiBulk(true)}
        onExport={() => {/* TODO */}}
        onClear={() => setMultiSelected(new Set())}
      />
    </div>
  )
}

function SortSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border bg-background px-2 py-1 text-xs outline-none"
    >
      <option value="name">名称</option>
      <option value="mtime">修改时间</option>
      <option value="size">大小</option>
    </select>
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