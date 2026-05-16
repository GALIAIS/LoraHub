import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  ChevronDown,
  ChevronRight,
  File as FileIcon,
  FileText,
  Folder,
  FolderOpen,
  HardDrive,
  Image as ImageIcon,
  Loader2,
  RefreshCw,
} from "lucide-react"
import { api, type FsEntry, type FsRoot } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface FileBrowserProps {
  /** Currently focused path (folder or file). Drives selection styling. */
  selectedPath: string | null
  /** Called when a folder is single-clicked. */
  onSelectFolder?: (path: string) => void
  /** Called when a file is single-clicked. */
  onSelectFile?: (entry: FsEntry) => void
  /** Called when a file is double-clicked or activated via Enter. */
  onOpenFile?: (entry: FsEntry) => void
  className?: string
}

/**
 * Lazy-loaded directory tree. Roots come from /api/fs/roots; each folder is
 * fetched on first expand and cached by react-query so re-expanding is free.
 * Designed to feel like marimo's file panel — no auto-expansion of remote
 * subtrees, no eager scanning.
 */
export function FileBrowser({
  selectedPath,
  onSelectFolder,
  onSelectFile,
  onOpenFile,
  className,
}: FileBrowserProps) {
  const rootsQuery = useQuery({
    queryKey: ["fs-roots"],
    queryFn: api.fsRoots,
    staleTime: 30_000,
  })

  return (
    <div className={cn("flex flex-col h-full min-h-0", className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/60">
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/85">
          文件浏览器
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="size-6"
          onClick={() => rootsQuery.refetch()}
          title="刷新"
        >
          <RefreshCw className="size-3" />
        </Button>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto px-1 py-1.5 text-[12px]">
        {rootsQuery.isLoading && (
          <div className="px-2 py-3 text-muted-foreground flex items-center gap-2">
            <Loader2 className="size-3 animate-spin" /> 加载中…
          </div>
        )}
        {rootsQuery.isError && (
          <div className="px-2 py-2 text-destructive text-[11px] font-mono break-all">
            {(rootsQuery.error as Error).message}
          </div>
        )}
        {rootsQuery.data?.roots.map((root) => (
          <RootNode
            key={root.path}
            root={root}
            selectedPath={selectedPath}
            onSelectFolder={onSelectFolder}
            onSelectFile={onSelectFile}
            onOpenFile={onOpenFile}
          />
        ))}
        {rootsQuery.data && rootsQuery.data.roots.length === 0 && (
          <div className="px-2 py-3 text-muted-foreground text-[11px]">
            未发现可浏览的根目录。
          </div>
        )}
        {rootsQuery.data && !rootsQuery.data.unrestricted && (
          <div className="mt-3 px-2 py-2 rounded-[3px] bg-muted/40 text-[10.5px] text-muted-foreground leading-relaxed">
            已限制在数据集根目录与训练 workspace。
            如需浏览其它路径，请在「设置 ▸ 概览」开启
            <code className="font-mono">allow_filesystem_browse</code>
            。
          </div>
        )}
      </div>
    </div>
  )
}

function RootNode({
  root,
  selectedPath,
  onSelectFolder,
  onSelectFile,
  onOpenFile,
}: {
  root: FsRoot
  selectedPath: string | null
  onSelectFolder?: (path: string) => void
  onSelectFile?: (entry: FsEntry) => void
  onOpenFile?: (entry: FsEntry) => void
}) {
  return (
    <DirectoryNode
      path={root.path}
      label={root.name}
      depth={0}
      icon={
        root.kind === "drive" ? (
          <HardDrive className="size-3.5 shrink-0 text-muted-foreground" />
        ) : null
      }
      defaultOpen={root.kind === "dataset_root"}
      selectedPath={selectedPath}
      onSelectFolder={onSelectFolder}
      onSelectFile={onSelectFile}
      onOpenFile={onOpenFile}
    />
  )
}

function DirectoryNode({
  path,
  label,
  depth,
  icon,
  defaultOpen = false,
  selectedPath,
  onSelectFolder,
  onSelectFile,
  onOpenFile,
}: {
  path: string
  label: string
  depth: number
  icon?: React.ReactNode
  defaultOpen?: boolean
  selectedPath: string | null
  onSelectFolder?: (path: string) => void
  onSelectFile?: (entry: FsEntry) => void
  onOpenFile?: (entry: FsEntry) => void
}) {
  const [open, setOpen] = useState(defaultOpen)
  const isSelected = selectedPath === path

  // Auto-open the chain leading to the selected path so the highlighted
  // entry stays visible across navigations.
  useEffect(() => {
    if (
      !open &&
      selectedPath &&
      (selectedPath === path || selectedPath.startsWith(path + "/") || selectedPath.startsWith(path + "\\"))
    ) {
      setOpen(true)
    }
  }, [selectedPath, path, open])

  const listing = useQuery({
    queryKey: ["fs-list", path],
    queryFn: () => api.fsList(path, false),
    enabled: open,
    staleTime: 5_000,
  })

  return (
    <div>
      <button
        type="button"
        onClick={() => {
          setOpen((v) => !v)
          onSelectFolder?.(path)
        }}
        className={cn(
          "w-full flex items-center gap-1 px-1.5 py-0.5 rounded-[3px] text-left hover:bg-muted/55 transition-colors",
          isSelected && "bg-primary/10 text-primary",
        )}
        style={{ paddingLeft: `${depth * 12 + 6}px` }}
        title={path}
      >
        {open ? (
          <ChevronDown className="size-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3 shrink-0 text-muted-foreground" />
        )}
        {icon ?? (
          open ? (
            <FolderOpen className="size-3.5 shrink-0 text-amber-500/85" />
          ) : (
            <Folder className="size-3.5 shrink-0 text-amber-500/85" />
          )
        )}
        <span className="truncate font-mono">{label}</span>
      </button>
      {open && (
        <div>
          {listing.isLoading && (
            <div
              className="text-[11px] text-muted-foreground py-0.5"
              style={{ paddingLeft: `${(depth + 1) * 12 + 18}px` }}
            >
              <Loader2 className="size-3 inline animate-spin mr-1" /> …
            </div>
          )}
          {listing.isError && (
            <div
              className="text-[11px] text-destructive py-0.5 break-all"
              style={{ paddingLeft: `${(depth + 1) * 12 + 18}px` }}
            >
              {(listing.error as Error).message}
            </div>
          )}
          {listing.data?.entries.map((entry) =>
            entry.is_dir ? (
              <DirectoryNode
                key={entry.path}
                path={entry.path}
                label={entry.name}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelectFolder={onSelectFolder}
                onSelectFile={onSelectFile}
                onOpenFile={onOpenFile}
              />
            ) : (
              <FileNode
                key={entry.path}
                entry={entry}
                depth={depth + 1}
                isSelected={selectedPath === entry.path}
                onSelectFile={onSelectFile}
                onOpenFile={onOpenFile}
              />
            ),
          )}
          {listing.data?.truncated && (
            <div
              className="text-[10px] text-amber-600 dark:text-amber-400 py-0.5"
              style={{ paddingLeft: `${(depth + 1) * 12 + 18}px` }}
            >
              已截断（仅显示前 5000 项）
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FileNode({
  entry,
  depth,
  isSelected,
  onSelectFile,
  onOpenFile,
}: {
  entry: FsEntry
  depth: number
  isSelected: boolean
  onSelectFile?: (entry: FsEntry) => void
  onOpenFile?: (entry: FsEntry) => void
}) {
  const Icon = useMemo(() => {
    if (entry.kind === "image") return ImageIcon
    if (entry.kind === "text") return FileText
    return FileIcon
  }, [entry.kind])
  return (
    <button
      type="button"
      onClick={() => onSelectFile?.(entry)}
      onDoubleClick={() => onOpenFile?.(entry)}
      onKeyDown={(e) => {
        if (e.key === "Enter") onOpenFile?.(entry)
      }}
      className={cn(
        "w-full flex items-center gap-1 px-1.5 py-0.5 rounded-[3px] text-left hover:bg-muted/55 transition-colors",
        isSelected && "bg-primary/10 text-primary",
      )}
      style={{ paddingLeft: `${depth * 12 + 18}px` }}
      title={entry.path}
    >
      <Icon
        className={cn(
          "size-3.5 shrink-0",
          entry.kind === "image" && "text-violet-500/85",
          entry.kind === "text" && "text-sky-500/85",
          entry.kind === "binary" && "text-muted-foreground",
        )}
      />
      <span className="truncate font-mono">{entry.name}</span>
    </button>
  )
}
