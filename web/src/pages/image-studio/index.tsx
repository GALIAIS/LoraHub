import { useCallback, useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  FlipHorizontal,
  FolderOpen,
  Heart,
  HelpCircle,
  Pencil,
  RotateCw,
  Save,
  Star,
  Trash2,
  Undo2,
  X,
} from "lucide-react"
import {
  imageStudioList,
  imageStudioGetImage,
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
  imageStudioDeleteOp,
  imageStudioDedupeScan,
  imageStudioDedupeClusters,
  imageStudioBatchDelete,
  type ImageStudioItem,
  type ImageStudioDetailItem,
  type DedupeCluster,
} from "@/lib/api"

export { ImageStudioPage }

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const path = params.get("path") || ""
  const page = Number(params.get("page") || "1")
  const sort = params.get("sort") || "name"
  const recursive = params.get("recursive") === "1"
  const view = params.get("view") || "grid"
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [inputPath, setInputPath] = useState(path)
  const [showHelp, setShowHelp] = useState(false)

  const setPage = (p: number) => {
    const next = new URLSearchParams(params)
    next.set("page", String(p))
    setParams(next)
  }

  const navigate = (newPath: string) => {
    const next = new URLSearchParams()
    next.set("path", newPath)
    setParams(next)
    setSelectedPath(null)
  }

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

  // Keyboard shortcuts
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
        case "Escape":
          setSelectedPath(null)
          setShowHelp(false)
          break
        case "?":
          e.preventDefault()
          setShowHelp((v) => !v)
          break
      }
    },
    [listQuery.data?.items, selectedPath],
  )

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])

  if (!path) {
    return <PathPrompt value={inputPath} onChange={setInputPath} onSubmit={navigate} />
  }

  const data = listQuery.data
  const totalPages = data ? Math.ceil(data.total / data.limit) : 0

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b px-4 py-2">
        <button
          type="button"
          onClick={() => navigate("")}
          className="rounded p-1 hover:bg-muted"
          title="切换文件夹"
        >
          <FolderOpen className="size-4" />
        </button>
        <span className="font-mono text-xs text-muted-foreground truncate flex-1">
          {path}
        </span>
        {data && (
          <span className="text-xs text-muted-foreground">
            {data.total} 张图片
          </span>
        )}
        <SortSelect value={sort} onChange={(s) => {
          const next = new URLSearchParams(params)
          next.set("sort", s)
          next.set("page", "1")
          setParams(next)
        }} />
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
          onClick={() => setShowHelp(true)}
          className="rounded p-1 text-muted-foreground hover:bg-muted"
          title="键盘快捷键 (?)"
        >
          <HelpCircle className="size-4" />
        </button>
      </div>

      {/* Help overlay */}
      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}

      {/* Main content */}
      {view === "duplicates" ? (
        <DuplicatesView path={path} recursive={recursive} />
      ) : (
      <div className="flex flex-1 min-h-0">
        {/* Image grid */}
        <div className="flex-1 overflow-y-auto p-3">
          {listQuery.isLoading && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              加载中...
            </div>
          )}
          {data && data.items.length === 0 && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              该目录下未找到图片
            </div>
          )}
          {data && data.items.length > 0 && (
            <>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2">
                {data.items.map((item) => (
                  <ImageTile
                    key={item.path}
                    item={item}
                    selected={item.path === selectedPath}
                    onClick={() => setSelectedPath(item.path)}
                  />
                ))}
              </div>
              {totalPages > 1 && (
                <Pagination page={page} total={totalPages} onChange={setPage} />
              )}
            </>
          )}
        </div>

        {/* Inspector */}
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
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Sub-components
// --------------------------------------------------------------------------- //

// --------------------------------------------------------------------------- //
// Duplicates view
// --------------------------------------------------------------------------- //

function DuplicatesView({ path, recursive }: { path: string; recursive: boolean }) {
  const queryClient = useQueryClient()
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())

  const scanMutation = useMutation({
    mutationFn: () => imageStudioDedupeScan({ path, recursive, algo: "phash64" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["image-studio", "clusters"] }),
  })

  const clustersQuery = useQuery({
    queryKey: ["image-studio", "clusters", path],
    queryFn: () => imageStudioDedupeClusters({ path, threshold: 10 }),
    enabled: !!path,
  })

  const deleteMutation = useMutation({
    mutationFn: () => imageStudioBatchDelete({ paths: Array.from(selectedPaths) }),
    onSuccess: () => {
      setSelectedPaths(new Set())
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    },
  })

  const clusters = clustersQuery.data?.clusters ?? []

  const togglePath = (p: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(p)) next.delete(p)
      else next.add(p)
      return next
    })
  }

  const selectAllSuggested = () => {
    const paths = new Set<string>()
    for (const c of clusters) {
      for (const m of c.members) {
        if (m.path !== c.suggestedKeep) paths.add(m.path)
      }
    }
    setSelectedPaths(paths)
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto p-4 gap-4">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending}
          className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
        >
          {scanMutation.isPending ? "扫描中..." : "扫描重复图片"}
        </button>
        {clusters.length > 0 && (
          <>
            <button
              type="button"
              onClick={selectAllSuggested}
              className="rounded border px-2 py-1 text-xs hover:bg-muted"
            >
              全选建议删除
            </button>
            <span className="text-xs text-muted-foreground">
              {clusters.length} 个聚类, 已选 {selectedPaths.size} 张
            </span>
            {selectedPaths.size > 0 && (
              <button
                type="button"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                className="rounded bg-destructive px-3 py-1.5 text-xs font-medium text-destructive-foreground disabled:opacity-50"
              >
                删除选中 ({selectedPaths.size})
              </button>
            )}
          </>
        )}
      </div>

      {clustersQuery.isLoading && (
        <p className="text-sm text-muted-foreground">加载聚类中...</p>
      )}

      {clusters.length === 0 && !clustersQuery.isLoading && (
        <p className="text-sm text-muted-foreground">
          未发现重复聚类。点击"扫描重复图片"开始分析。
        </p>
      )}

      {clusters.map((cluster) => (
        <ClusterCard
          key={cluster.id}
          cluster={cluster}
          selectedPaths={selectedPaths}
          onToggle={togglePath}
        />
      ))}
    </div>
  )
}

function ClusterCard({
  cluster,
  selectedPaths,
  onToggle,
}: {
  cluster: DedupeCluster
  selectedPaths: Set<string>
  onToggle: (path: string) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-md border">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-muted/50"
      >
        <span className="font-medium">{cluster.id}</span>
        <span className="text-xs text-muted-foreground">
          {cluster.members.length} 张图片
        </span>
        <span className="ml-auto text-xs">{expanded ? "▼" : "▶"}</span>
      </button>
      {expanded && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2 border-t p-3">
          {cluster.members.map((m) => {
            const isKeep = m.path === cluster.suggestedKeep
            const isSelected = selectedPaths.has(m.path)
            return (
              <div
                key={m.path}
                className={`relative flex flex-col overflow-hidden rounded border ${
                  isKeep ? "border-green-500/50 ring-1 ring-green-500/30" : ""
                } ${isSelected ? "border-destructive ring-1 ring-destructive/30" : ""}`}
              >
                <div className="aspect-square overflow-hidden bg-muted">
                  <img
                    src={`/api/datasets/thumb?path=${encodeURIComponent(m.path)}&size=128`}
                    alt=""
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="flex items-center gap-1 px-1.5 py-1">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggle(m.path)}
                    className="size-3"
                  />
                  <span className="flex-1 truncate text-[10px]">
                    {m.path.split(/[/\\]/).pop()}
                  </span>
                </div>
                {isKeep && (
                  <div className="absolute right-1 top-1 rounded bg-green-600/80 px-1 py-0.5 text-[9px] font-medium text-white">
                    保留
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function PathPrompt({
  value,
  onChange,
  onSubmit,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: (v: string) => void
}) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="flex w-full max-w-lg flex-col gap-4">
        <h1 className="text-lg font-semibold">图像工作台</h1>
        <p className="text-sm text-muted-foreground">
          输入数据集文件夹路径，开始处理训练图片。
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (value.trim()) onSubmit(value.trim())
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="数据集路径，如 /path/to/dataset"
            className="flex-1 rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/30"
          />
          <button
            type="submit"
            disabled={!value.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            打开
          </button>
        </form>
      </div>
    </div>
  )
}

function ImageTile({
  item,
  selected,
  onClick,
}: {
  item: ImageStudioItem
  selected: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group relative flex flex-col overflow-hidden rounded-md border transition-colors ${
        selected
          ? "border-primary ring-2 ring-primary/30"
          : "border-border hover:border-muted-foreground/40"
      }`}
    >
      <div className="aspect-square w-full overflow-hidden bg-muted">
        <img
          src={item.thumbUrl}
          alt={item.name}
          loading="lazy"
          className="h-full w-full object-cover"
        />
      </div>
      <div className="flex items-center gap-1 px-1.5 py-1">
        <span className="flex-1 truncate text-left text-[11px]">{item.name}</span>
        {item.annotation?.favorite && (
          <Heart className="size-3 fill-rose-500 text-rose-500" />
        )}
        {item.annotation?.softDeleted && (
          <Trash2 className="size-3 text-muted-foreground" />
        )}
      </div>
      {!item.captionExists && (
        <div className="absolute right-1 top-1 rounded bg-amber-500/80 px-1 py-0.5 text-[9px] font-medium text-white">
          无描述
        </div>
      )}
    </button>
  )
}

function Inspector({
  detail,
  loading,
  path,
  onClose,
}: {
  detail: ImageStudioDetailItem | null
  loading: boolean
  path: string
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [editingCaption, setEditingCaption] = useState(false)
  const [captionDraft, setCaptionDraft] = useState("")

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["image-studio"] })

  const favMutation = useMutation({
    mutationFn: (fav: boolean) =>
      imageStudioSaveAnnotation({ path, favorite: fav }),
    onSuccess: invalidate,
  })

  const addOpMutation = useMutation({
    mutationFn: (body: { op: string; payload?: Record<string, unknown> }) =>
      imageStudioAddOp({ path, ...body }),
    onSuccess: invalidate,
  })

  const deleteOpMutation = useMutation({
    mutationFn: (opId: string) => imageStudioDeleteOp(opId),
    onSuccess: invalidate,
  })

  const applyMutation = useMutation({
    mutationFn: () => imageStudioApplyOps(path),
    onSuccess: invalidate,
  })

  const startCaptionEdit = () => {
    setCaptionDraft(detail?.caption || "")
    setEditingCaption(true)
  }

  const saveCaption = () => {
    addOpMutation.mutate({ op: "replace_caption", payload: { caption: captionDraft } })
    setEditingCaption(false)
  }

  return (
    <aside className="w-[22rem] shrink-0 overflow-y-auto border-l bg-background p-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium truncate">{detail?.name ?? "..."}</h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-muted-foreground hover:bg-muted"
        >
          &times;
        </button>
      </div>

      {loading && <p className="text-xs text-muted-foreground">加载中...</p>}

      {detail && (
        <div className="flex flex-col gap-3">
          {/* Preview */}
          <div className="overflow-hidden rounded-md border">
            <img
              src={`/api/datasets/thumb?path=${encodeURIComponent(detail.path)}&size=512`}
              alt={detail.name}
              className="w-full"
            />
          </div>

          {/* Meta */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <span className="text-muted-foreground">文件大小</span>
            <span>{formatBytes(detail.bytes)}</span>
            {detail.width && detail.height && (
              <>
                <span className="text-muted-foreground">尺寸</span>
                <span>{detail.width} x {detail.height}</span>
              </>
            )}
          </div>

          {/* Caption */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">描述</span>
              {!editingCaption && (
                <button
                  type="button"
                  onClick={startCaptionEdit}
                  className="rounded p-0.5 text-muted-foreground hover:bg-muted"
                >
                  <Pencil className="size-3" />
                </button>
              )}
            </div>
            {editingCaption ? (
              <div className="flex flex-col gap-1.5">
                <textarea
                  value={captionDraft}
                  onChange={(e) => setCaptionDraft(e.target.value)}
                  rows={4}
                  className="w-full rounded border bg-background px-2 py-1.5 text-xs outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 resize-y"
                />
                <div className="flex gap-1.5">
                  <button
                    type="button"
                    onClick={saveCaption}
                    className="flex items-center gap-1 rounded bg-primary px-2 py-1 text-xs text-primary-foreground"
                  >
                    <Save className="size-3" /> 保存
                  </button>
                  <button
                    type="button"
                    onClick={() => setEditingCaption(false)}
                    className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
                  >
                    <X className="size-3" /> 取消
                  </button>
                </div>
              </div>
            ) : detail.caption ? (
              <p className="rounded bg-muted/50 p-2 text-xs leading-relaxed">
                {detail.caption}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">无描述文件</p>
            )}
          </div>

          {/* AI annotation */}
          {detail.annotation?.aiQualityLabel && (
            <div className="flex items-center gap-2">
              <Star className="size-3.5 text-amber-500" />
              <span className="text-xs">
                质量: {detail.annotation.aiQualityLabel}
                {detail.annotation.aiQualityScore != null &&
                  ` (${(detail.annotation.aiQualityScore * 100).toFixed(0)}%)`}
              </span>
            </div>
          )}

          {/* Edit actions */}
          <div className="flex flex-wrap gap-1.5 pt-2 border-t">
            <button
              type="button"
              onClick={() => favMutation.mutate(!detail.annotation?.favorite)}
              className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors ${
                detail.annotation?.favorite
                  ? "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400"
                  : "hover:bg-muted"
              }`}
            >
              <Heart className="size-3" />
              {detail.annotation?.favorite ? "取消收藏" : "收藏"}
            </button>
            <button
              type="button"
              onClick={() => addOpMutation.mutate({ op: "rotate", payload: { degrees: 90 } })}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
            >
              <RotateCw className="size-3" /> 旋转
            </button>
            <button
              type="button"
              onClick={() => addOpMutation.mutate({ op: "flip", payload: { direction: "horizontal" } })}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
            >
              <FlipHorizontal className="size-3" /> 翻转
            </button>
            <button
              type="button"
              onClick={() => addOpMutation.mutate({ op: "delete", payload: {} })}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="size-3" /> 删除
            </button>
          </div>

          {/* Pending ops */}
          {detail.pendingOps.length > 0 && (
            <div className="flex flex-col gap-1.5 pt-2 border-t">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">
                  待处理 ({detail.pendingOps.length})
                </span>
                <button
                  type="button"
                  onClick={() => applyMutation.mutate()}
                  disabled={applyMutation.isPending}
                  className="flex items-center gap-1 rounded bg-primary px-2 py-0.5 text-[11px] text-primary-foreground disabled:opacity-50"
                >
                  全部应用
                </button>
              </div>
              {detail.pendingOps.map((op) => (
                <div
                  key={op.id}
                  className="flex items-center gap-2 rounded bg-muted/50 px-2 py-1 text-xs"
                >
                  <RotateCw className="size-3 shrink-0" />
                  <span className="flex-1">{op.op}</span>
                  <button
                    type="button"
                    onClick={() => deleteOpMutation.mutate(op.id)}
                    className="rounded p-0.5 text-muted-foreground hover:text-destructive"
                  >
                    <Undo2 className="size-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

function SortSelect({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
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

function HelpOverlay({ onClose }: { onClose: () => void }) {
  const shortcuts = [
    { key: "j / k", desc: "在网格中向下/向上导航" },
    { key: "Escape", desc: "关闭检查器/帮助" },
    { key: "?", desc: "切换帮助面板" },
  ]

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="w-80 rounded-lg border bg-popover p-4 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">键盘快捷键</h3>
          <button type="button" onClick={onClose} className="rounded p-1 hover:bg-muted">
            <X className="size-4" />
          </button>
        </div>
        <div className="flex flex-col gap-1.5">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between text-xs">
              <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {s.key}
              </kbd>
              <span className="text-muted-foreground">{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
