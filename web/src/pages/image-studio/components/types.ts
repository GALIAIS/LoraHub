import type { ImageStudioItem } from "@/lib/api"

// Filter state for the filter panel
export interface FilterState {
  caption: "all" | "has" | "missing"
  quality: "all" | "star_5" | "star_4" | "star_3" | "star_2" | "star_1" | "unrated" | "favorite"
  aspect: "all" | "landscape" | "portrait" | "square"
  /** Free-text search over caption + filename. Empty string disables. */
  search: string
  /** Subdir prefix (relative to dataset root). Empty string = no filter. */
  subdir: string
}

export const defaultFilters: FilterState = {
  caption: "all",
  quality: "all",
  aspect: "all",
  search: "",
  subdir: "",
}

// Multi-select state
export interface SelectionState {
  mode: "single" | "multi"
  selected: Set<string>
}

// AI Bulk operation tab
export type AiBulkTab =
  | "smart-caption"
  | "quality-score"
  | "wd14"
  | "trigger-words"

/**
 * Saved filter preset. Persists to localStorage so the user can name
 * "缺描述 + 横向" once and recall it across sessions.
 */
export interface FilterPreset {
  id: string
  name: string
  filters: FilterState
}

export const FILTER_PRESET_STORAGE_KEY = "lorahub.image-studio.filter-presets"

export function loadFilterPresets(): FilterPreset[] {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(FILTER_PRESET_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (p): p is FilterPreset =>
        typeof p === "object" &&
        p !== null &&
        typeof p.id === "string" &&
        typeof p.name === "string" &&
        typeof p.filters === "object",
    )
  } catch {
    return []
  }
}

export function saveFilterPresets(presets: FilterPreset[]): void {
  if (typeof window === "undefined") return
  window.localStorage.setItem(
    FILTER_PRESET_STORAGE_KEY,
    JSON.stringify(presets),
  )
}

// Apply client-side filters to items
export function applyFilters(
  items: ImageStudioItem[],
  filters: FilterState,
): ImageStudioItem[] {
  const search = filters.search.trim().toLowerCase()
  const subdir = filters.subdir.trim().replace(/^[\\/]+|[\\/]+$/g, "")
  return items.filter((item) => {
    // Subdir prefix — relativePath is "<subdir>/<file>" form, so we
    // gate on the relativePath containing the prefix as the leading
    // path segment.
    if (subdir) {
      const rel = item.relativePath.replace(/\\/g, "/")
      const prefix = subdir.replace(/\\/g, "/") + "/"
      if (!rel.startsWith(prefix)) return false
    }

    // Free-text search across caption, filename, relativePath.
    if (search) {
      const haystack = [
        item.name,
        item.relativePath,
        item.caption ?? "",
        item.annotation?.aiCaption ?? "",
        item.annotation?.userNotes ?? "",
      ]
        .join(" ")
        .toLowerCase()
      if (!haystack.includes(search)) return false
    }

    // Caption filter
    if (filters.caption === "has" && !item.captionExists) return false
    if (filters.caption === "missing" && item.captionExists) return false

    // Quality filter
    if (filters.quality !== "all") {
      const userQuality = normalizeManualQuality(item.annotation?.userQualityLabel)
      if (filters.quality === "favorite") {
        if (!item.annotation?.favorite) return false
      } else if (filters.quality === "unrated") {
        if (item.annotation?.aiQualityLabel || item.annotation?.userQualityLabel) return false
      } else {
        if (userQuality !== filters.quality) return false
      }
    }

    // Aspect ratio filter
    if (filters.aspect !== "all" && item.width && item.height) {
      const ratio = item.width / item.height
      if (filters.aspect === "landscape" && ratio <= 1.05) return false
      if (filters.aspect === "portrait" && ratio >= 0.95) return false
      if (filters.aspect === "square" && (ratio < 0.95 || ratio > 1.05)) return false
    }

    return true
  })
}

function normalizeManualQuality(value: string | null | undefined): FilterState["quality"] | null {
  if (!value) return null
  if (/^star_[1-5]$/.test(value)) return value as FilterState["quality"]
  if (value === "good") return "star_5"
  if (value === "ok" || value === "medium") return "star_3"
  if (value === "bad") return "star_1"
  return null
}

/**
 * Build a subdir tree from a flat list of items.
 * Each node carries the prefix path and the count of images directly
 * (or recursively) underneath it.
 */
export interface SubdirNode {
  /** Full prefix path relative to dataset root, e.g. "anime/style/a". */
  prefix: string
  /** Last segment, used as the display label. */
  label: string
  /** Number of image items whose relativePath starts with this prefix. */
  count: number
  children: SubdirNode[]
}

export function buildSubdirTree(items: ImageStudioItem[]): SubdirNode[] {
  // Map prefix -> { count, children: Set<string> }
  const counts = new Map<string, number>()
  const childMap = new Map<string, Set<string>>()
  const roots = new Set<string>()

  for (const item of items) {
    const rel = item.relativePath.replace(/\\/g, "/")
    const parts = rel.split("/")
    if (parts.length <= 1) continue // top-level files don't contribute to the tree
    const dirParts = parts.slice(0, -1)
    let prefix = ""
    for (let i = 0; i < dirParts.length; i++) {
      const seg = dirParts[i]
      const next = prefix ? `${prefix}/${seg}` : seg
      counts.set(next, (counts.get(next) ?? 0) + 1)
      if (i === 0) {
        roots.add(next)
      } else {
        const set = childMap.get(prefix) ?? new Set<string>()
        set.add(next)
        childMap.set(prefix, set)
      }
      prefix = next
    }
  }

  const buildNode = (prefix: string): SubdirNode => {
    const label = prefix.split("/").pop() ?? prefix
    const childSet = childMap.get(prefix)
    const children = childSet
      ? Array.from(childSet)
          .sort()
          .map(buildNode)
      : []
    return {
      prefix,
      label,
      count: counts.get(prefix) ?? 0,
      children,
    }
  }

  return Array.from(roots).sort().map(buildNode)
}
