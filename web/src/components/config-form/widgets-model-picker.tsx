/**
 * Combobox-style model picker.
 *
 * Replaces the bare PathInput for fields that point at a .safetensors
 * file living under the project's models/ folder. Hitting the
 * dropdown triggers a fresh GET /api/models/scan call (manual scan
 * — never a startup-time auto-walk so users see exactly the files
 * they have right now), and the resulting list is filtered live as
 * the user types. Free-form text input is still allowed: typing a
 * path that doesn't appear in the scan result keeps that exact
 * value, so users with models outside the repo aren't trapped.
 */
import { useEffect, useMemo, useRef, useState } from "react"
import { ChevronsUpDown, FolderSearch, RefreshCw } from "lucide-react"
import { api, type ScannedModel } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

export interface ModelPathPickerProps {
  value: string
  onChange: (next: string) => void
  placeholder?: string
  /**
   * Pin the scan to a sub-path of the workspace (e.g. ``vae`` or
   * ``circlestone-labs__Anima``) when the field is known to want only
   * one component family. Falls back to ``models/`` when omitted.
   */
  rootHint?: string | undefined
  className?: string
}

function _formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(0)} MiB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GiB`
}

export function ModelPathPicker({
  value,
  onChange,
  placeholder,
  rootHint,
  className,
}: ModelPathPickerProps) {
  const [open, setOpen] = useState(false)
  const [files, setFiles] = useState<ScannedModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scannedRoot, setScannedRoot] = useState<string | null>(null)
  const [filter, setFilter] = useState("")
  const dropdownRef = useRef<HTMLDivElement | null>(null)

  async function rescan() {
    setLoading(true)
    setError(null)
    try {
      const resp = await api.scanModels(rootHint)
      setFiles(resp.files)
      setScannedRoot(resp.root)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  // Auto-scan once when the dropdown opens for the first time (lazy —
  // a full scan can take ~100ms even with caching, no point doing it
  // before the user actually clicks).
  useEffect(() => {
    if (open && files.length === 0 && !loading && error === null) {
      void rescan()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Close the dropdown when clicking outside.
  useEffect(() => {
    if (!open) return
    function onDoc(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", onDoc)
    return () => document.removeEventListener("mousedown", onDoc)
  }, [open])

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return files
    return files.filter(
      (f) =>
        f.relative_path.toLowerCase().includes(q) ||
        f.name.toLowerCase().includes(q),
    )
  }, [files, filter])

  return (
    <div className={cn("relative w-full max-w-2xl", className)} ref={dropdownRef}>
      <div className="flex items-center gap-1.5">
        <Input
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setOpen(true)}
          placeholder={placeholder ?? "models/<vendor>/<file>.safetensors"}
          className="font-mono text-xs flex-1"
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => setOpen((o) => !o)}
          title="从 models/ 中选择"
          className="h-9 w-9 shrink-0"
        >
          <ChevronsUpDown className="size-3.5" />
        </Button>
      </div>

      {open && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 rounded-[5px] border border-border/60 bg-popover shadow-lg max-h-80 overflow-hidden flex flex-col">
          <div className="flex items-center gap-1.5 border-b border-border/40 px-2 py-1.5">
            <FolderSearch className="size-3.5 text-muted-foreground shrink-0" />
            <Input
              autoFocus
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="搜索文件名 / 子目录…"
              className="h-7 font-mono text-[11px] border-none focus-visible:ring-0 px-0"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={rescan}
              disabled={loading}
              className="h-7 px-2 text-[10px] gap-1"
            >
              <RefreshCw
                className={cn("size-3", loading && "animate-spin")}
              />
              重扫
            </Button>
          </div>

          {scannedRoot && (
            <div className="px-2 py-1 text-[10px] text-muted-foreground/70 font-mono truncate border-b border-border/30">
              扫描根：{scannedRoot}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">
                扫描中…
              </div>
            ) : error ? (
              <div className="px-3 py-3 text-[11px] text-destructive">
                扫描失败：{error}
              </div>
            ) : filtered.length === 0 ? (
              <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">
                {files.length === 0 ? "models/ 下没有找到模型文件。" : "没有匹配的结果。"}
              </div>
            ) : (
              <ul>
                {filtered.map((f) => {
                  // Backend ``GET /api/models/scan`` returns paths
                  // relative to the resolved scan root (always
                  // ``<project>/models``). The config yaml needs the
                  // ``models/`` prefix because lifecycle resolves
                  // config paths against ``Path.cwd()`` (the project
                  // root), not against ``<project>/models``. Without
                  // the prefix the path resolves to the wrong place
                  // and training fails with "checkpoint not found".
                  const pickedPath = `models/${f.relative_path}`
                  const active =
                    value === pickedPath ||
                    value === f.path ||
                    value === f.relative_path
                  return (
                    <li key={f.path}>
                      <button
                        type="button"
                        onClick={() => {
                          onChange(pickedPath)
                          setOpen(false)
                        }}
                        className={cn(
                          "w-full text-left px-2.5 py-1.5 hover:bg-muted/50",
                          active && "bg-primary/10",
                        )}
                      >
                        <div className="font-mono text-[11px] truncate">
                          {f.relative_path}
                        </div>
                        <div className="text-[10px] text-muted-foreground tabular-nums">
                          {_formatBytes(f.size_bytes)}
                        </div>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
