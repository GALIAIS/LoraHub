import { useEffect, useMemo, useState } from "react"
import { ChevronDown, ChevronRight, FolderOpen, Save, Search, Trash2, X } from "lucide-react"
import { toast } from "sonner"
import type { ImageStudioItem } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import {
  buildSubdirTree,
  loadFilterPresets,
  saveFilterPresets,
  type FilterPreset,
  type FilterState,
  type SubdirNode,
} from "./types"

interface FilterPanelProps {
  filters: FilterState
  onChange: (filters: FilterState) => void
  onClose: () => void
  /** Used to derive the subdir tree shown in this panel. */
  items?: ImageStudioItem[]
  /** Reset filters to defaults — kept here so the page owns the default. */
  onReset?: () => void
}

export function FilterPanel({
  filters,
  onChange,
  onClose,
  items = [],
  onReset,
}: FilterPanelProps) {
  const set = <K extends keyof FilterState>(key: K, value: FilterState[K]) =>
    onChange({ ...filters, [key]: value })

  const tree = useMemo(() => buildSubdirTree(items), [items])
  const hasSubdirs = tree.length > 0

  const [presets, setPresets] = useState<FilterPreset[]>([])
  const [showPresetForm, setShowPresetForm] = useState(false)
  const [presetName, setPresetName] = useState("")

  useEffect(() => {
    setPresets(loadFilterPresets())
  }, [])

  const persist = (next: FilterPreset[]) => {
    setPresets(next)
    saveFilterPresets(next)
  }

  const savePreset = () => {
    const name = presetName.trim()
    if (!name) return
    const next: FilterPreset[] = [
      ...presets.filter((p) => p.name !== name),
      { id: cryptoRandomId(), name, filters },
    ]
    persist(next)
    setShowPresetForm(false)
    setPresetName("")
    toast.success(`已保存预设 "${name}"`)
  }

  const applyPreset = (preset: FilterPreset) => {
    onChange(preset.filters)
  }

  const removePreset = (id: string) => {
    persist(presets.filter((p) => p.id !== id))
  }

  return (
    <aside className="shiro-page-aside w-60 shrink-0 overflow-y-auto p-3">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold">筛选</span>
        <div className="flex items-center gap-1">
          {onReset && (
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={onReset}
              title="重置筛选"
            >
              <X className="size-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={onClose}
            title="收起筛选面板"
          >
            <ChevronRight className="size-3" />
          </Button>
        </div>
      </div>

      <FilterGroup label="关键字">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3 text-muted-foreground/70" />
          <Input
            value={filters.search}
            onChange={(e) => set("search", e.target.value)}
            placeholder="搜描述 / 文件名"
            className="h-7 pl-7 text-[11px]"
          />
        </div>
      </FilterGroup>

      <FilterGroup label="描述覆盖">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "has", label: "有描述" },
            { value: "missing", label: "缺描述" },
          ]}
          value={filters.caption}
          onChange={(v) => set("caption", v as FilterState["caption"])}
        />
      </FilterGroup>

      <FilterGroup label="质量评级">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "good", label: "优" },
            { value: "medium", label: "中" },
            { value: "bad", label: "差" },
            { value: "unrated", label: "未评" },
            { value: "favorite", label: "收藏" },
          ]}
          value={filters.quality}
          onChange={(v) => set("quality", v as FilterState["quality"])}
        />
      </FilterGroup>

      <FilterGroup label="宽高比">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "landscape", label: "横向" },
            { value: "portrait", label: "纵向" },
            { value: "square", label: "方形" },
          ]}
          value={filters.aspect}
          onChange={(v) => set("aspect", v as FilterState["aspect"])}
        />
      </FilterGroup>

      {hasSubdirs && (
        <FilterGroup label="子目录">
          <SubdirTree
            tree={tree}
            selected={filters.subdir}
            onSelect={(prefix) => set("subdir", prefix)}
          />
        </FilterGroup>
      )}

      <div className="mt-4 pt-3 border-t border-border/40 space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">
            预设
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-[10px]"
            onClick={() => {
              setShowPresetForm((v) => !v)
              setPresetName("")
            }}
          >
            <Save className="size-3" />
            保存当前
          </Button>
        </div>
        {showPresetForm && (
          <div className="flex items-center gap-1">
            <Input
              autoFocus
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault()
                  savePreset()
                } else if (e.key === "Escape") {
                  setShowPresetForm(false)
                  setPresetName("")
                }
              }}
              placeholder="预设名"
              className="h-7 text-[11px]"
            />
            <Button
              size="sm"
              className="h-7 text-[10px] shrink-0"
              onClick={savePreset}
              disabled={!presetName.trim()}
            >
              保存
            </Button>
          </div>
        )}
        {presets.length === 0 ? (
          <p className="text-[10.5px] text-muted-foreground/70 italic">
            尚无预设。先调好筛选后&quot;保存当前&quot;。
          </p>
        ) : (
          <ul className="space-y-0.5">
            {presets.map((p) => (
              <li
                key={p.id}
                className="flex items-center gap-1 group/preset"
              >
                <button
                  type="button"
                  onClick={() => applyPreset(p)}
                  className="flex-1 truncate rounded px-1.5 py-0.5 text-left text-[11px] hover:bg-muted/60 transition-colors"
                  title={summarizePreset(p.filters)}
                >
                  {p.name}
                </button>
                <button
                  type="button"
                  onClick={() => removePreset(p.id)}
                  className="rounded p-0.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive opacity-0 group-hover/preset:opacity-100 transition-opacity"
                  aria-label={`删除预设 ${p.name}`}
                >
                  <Trash2 className="size-3" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      <div className="mt-1">{children}</div>
    </div>
  )
}

function FilterRadio({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "rounded px-1.5 py-0.5 text-[11px] transition-colors",
            value === opt.value
              ? "bg-primary text-primary-foreground"
              : "bg-muted/50 text-muted-foreground hover:bg-muted",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function SubdirTree({
  tree,
  selected,
  onSelect,
}: {
  tree: SubdirNode[]
  selected: string
  onSelect: (prefix: string) => void
}) {
  return (
    <div className="space-y-0.5">
      <button
        type="button"
        onClick={() => onSelect("")}
        className={cn(
          "flex items-center gap-1 w-full rounded px-1 py-0.5 text-[11px] transition-colors",
          selected === ""
            ? "bg-primary/10 text-primary font-medium"
            : "text-muted-foreground hover:bg-muted/60",
        )}
      >
        <FolderOpen className="size-3" />
        <span>全部</span>
      </button>
      {tree.map((node) => (
        <SubdirNodeRow
          key={node.prefix}
          node={node}
          depth={0}
          selected={selected}
          onSelect={onSelect}
        />
      ))}
    </div>
  )
}

function SubdirNodeRow({
  node,
  depth,
  selected,
  onSelect,
}: {
  node: SubdirNode
  depth: number
  selected: string
  onSelect: (prefix: string) => void
}) {
  const [expanded, setExpanded] = useState(depth === 0)
  const hasChildren = node.children.length > 0
  const active = selected === node.prefix
  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-0.5 rounded text-[11px] transition-colors",
          active
            ? "bg-primary/10 text-primary font-medium"
            : "text-muted-foreground hover:bg-muted/60",
        )}
        style={{ paddingLeft: `${depth * 10}px` }}
      >
        {hasChildren ? (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="rounded p-0.5"
            aria-label={expanded ? "收起" : "展开"}
          >
            {expanded ? (
              <ChevronDown className="size-3" />
            ) : (
              <ChevronRight className="size-3" />
            )}
          </button>
        ) : (
          <span className="inline-block size-4 shrink-0" aria-hidden />
        )}
        <button
          type="button"
          onClick={() => onSelect(node.prefix)}
          className="flex flex-1 items-center gap-1 truncate py-0.5 text-left"
          title={node.prefix}
        >
          <span className="truncate">{node.label}</span>
          <span className="ml-auto pr-1 text-[10px] tabular-nums text-muted-foreground/70">
            {node.count}
          </span>
        </button>
      </div>
      {expanded && hasChildren && (
        <div>
          {node.children.map((child) => (
            <SubdirNodeRow
              key={child.prefix}
              node={child}
              depth={depth + 1}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function summarizePreset(state: FilterState): string {
  const parts: string[] = []
  if (state.search) parts.push(`搜="${state.search}"`)
  if (state.caption !== "all") parts.push(`描述=${state.caption}`)
  if (state.quality !== "all") parts.push(`质量=${state.quality}`)
  if (state.aspect !== "all") parts.push(`宽高比=${state.aspect}`)
  if (state.subdir) parts.push(`子目录=${state.subdir}`)
  return parts.length === 0 ? "默认筛选" : parts.join(" · ")
}

function cryptoRandomId(): string {
  // crypto.randomUUID exists in modern browsers; fall back to a
  // timestamp + random suffix that's good enough for a localStorage
  // key.
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `p-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}
