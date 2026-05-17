import type { ImageStudioItem } from "@/lib/api"

// Filter state for the filter panel
export interface FilterState {
  caption: "all" | "has" | "missing"
  quality: "all" | "good" | "medium" | "bad" | "unrated" | "favorite"
  aspect: "all" | "landscape" | "portrait" | "square"
}

export const defaultFilters: FilterState = {
  caption: "all",
  quality: "all",
  aspect: "all",
}

// Multi-select state
export interface SelectionState {
  mode: "single" | "multi"
  selected: Set<string>
}

// AI Bulk operation tab
export type AiBulkTab =
  | "smart-caption"
  | "vlm-caption"
  | "quality-score"
  | "wd14"
  | "trigger-words"

// Apply client-side filters to items
export function applyFilters(
  items: ImageStudioItem[],
  filters: FilterState,
): ImageStudioItem[] {
  return items.filter((item) => {
    // Caption filter
    if (filters.caption === "has" && !item.captionExists) return false
    if (filters.caption === "missing" && item.captionExists) return false

    // Quality filter
    if (filters.quality !== "all") {
      const label = item.annotation?.aiQualityLabel ?? item.annotation?.userQualityLabel
      if (filters.quality === "favorite") {
        if (!item.annotation?.favorite) return false
      } else if (filters.quality === "unrated") {
        if (label) return false
      } else {
        if (label !== filters.quality) return false
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
