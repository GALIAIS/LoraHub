import { useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  FolderOpen,
  Heart,
  RotateCw,
  Star,
  Trash2,
} from "lucide-react"
import {
  imageStudioList,
  imageStudioGetImage,
  imageStudioSaveAnnotation,
  type ImageStudioItem,
  type ImageStudioDetailItem,
} from "@/lib/api"

export { ImageStudioPage }

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const path = params.get("path") || ""
  const page = Number(params.get("page") || "1")
  const sort = params.get("sort") || "name"
  const recursive = params.get("recursive") === "1"
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [inputPath, setInputPath] = useState(path)

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
          title="Change folder"
        >
          <FolderOpen className="size-4" />
        </button>
        <span className="font-mono text-xs text-muted-foreground truncate flex-1">
          {path}
        </span>
        {data && (
          <span className="text-xs text-muted-foreground">
            {data.total} images
          </span>
        )}
        <SortSelect value={sort} onChange={(s) => {
          const next = new URLSearchParams(params)
          next.set("sort", s)
          next.set("page", "1")
          setParams(next)
        }} />
      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0">
        {/* Image grid */}
        <div className="flex-1 overflow-y-auto p-3">
          {listQuery.isLoading && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              Loading...
            </div>
          )}
          {data && data.items.length === 0 && (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              No images found in this directory.
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
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Sub-components
// --------------------------------------------------------------------------- //

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
        <h1 className="text-lg font-semibold">Image Studio</h1>
        <p className="text-sm text-muted-foreground">
          Enter a dataset folder path to begin working on your training images.
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
            placeholder="/path/to/dataset"
            className="flex-1 rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/30"
          />
          <button
            type="submit"
            disabled={!value.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Open
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
          No caption
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
  const favMutation = useMutation({
    mutationFn: (fav: boolean) =>
      imageStudioSaveAnnotation({ path, favorite: fav }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    },
  })

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

      {loading && <p className="text-xs text-muted-foreground">Loading...</p>}

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
            <span className="text-muted-foreground">Size</span>
            <span>{formatBytes(detail.bytes)}</span>
            {detail.width && detail.height && (
              <>
                <span className="text-muted-foreground">Dimensions</span>
                <span>{detail.width} x {detail.height}</span>
              </>
            )}
          </div>

          {/* Caption */}
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium">Caption</span>
            {detail.caption ? (
              <p className="rounded bg-muted/50 p-2 text-xs leading-relaxed">
                {detail.caption}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">No caption file</p>
            )}
          </div>

          {/* AI annotation */}
          {detail.annotation?.aiQualityLabel && (
            <div className="flex items-center gap-2">
              <Star className="size-3.5 text-amber-500" />
              <span className="text-xs">
                Quality: {detail.annotation.aiQualityLabel}
                {detail.annotation.aiQualityScore != null &&
                  ` (${(detail.annotation.aiQualityScore * 100).toFixed(0)}%)`}
              </span>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2 border-t">
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
              {detail.annotation?.favorite ? "Unfav" : "Fav"}
            </button>
          </div>

          {/* Pending ops */}
          {detail.pendingOps.length > 0 && (
            <div className="flex flex-col gap-1 pt-2 border-t">
              <span className="text-xs font-medium">
                Pending ops ({detail.pendingOps.length})
              </span>
              {detail.pendingOps.map((op) => (
                <div
                  key={op.id}
                  className="flex items-center gap-2 rounded bg-muted/50 px-2 py-1 text-xs"
                >
                  <RotateCw className="size-3" />
                  <span>{op.op}</span>
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
      <option value="name">Name</option>
      <option value="mtime">Modified</option>
      <option value="size">Size</option>
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
        Prev
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
        Next
      </button>
    </div>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
