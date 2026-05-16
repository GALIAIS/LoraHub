import { useMemo } from "react"
import { ChevronRight } from "lucide-react"

export interface PathBarProps {
  /** Path the bar is anchored at (the dataset root). */
  path: string
  /** Called when a breadcrumb segment is chosen. */
  onNavigate: (path: string) => void
}

/**
 * Breadcrumb navigation for the dataset scan card. Click any prefix
 * segment to scan that ancestor directory instead.
 */
export function PathBar({ path, onNavigate }: PathBarProps) {
  const segments = useMemo(() => splitSegments(path), [path])

  return (
    <nav className="flex items-center flex-wrap gap-0.5 text-[12px] font-mono text-muted-foreground min-w-0">
      {segments.map((seg, idx) => {
        const isLast = idx === segments.length - 1
        return (
          <div key={`${seg.absolute}-${idx}`} className="flex items-center gap-0.5 min-w-0">
            {idx > 0 && (
              <ChevronRight className="size-3 shrink-0 text-muted-foreground/60" />
            )}
            {isLast ? (
              <span className="truncate text-foreground" title={seg.absolute}>
                {seg.label}
              </span>
            ) : (
              <button
                type="button"
                onClick={() => onNavigate(seg.absolute)}
                className="px-1 py-0.5 rounded-[2px] hover:bg-muted/55 hover:text-foreground transition-colors truncate max-w-[10rem]"
                title={seg.absolute}
              >
                {seg.label}
              </button>
            )}
          </div>
        )
      })}
    </nav>
  )
}

/**
 * Split a path into breadcrumb segments where each segment carries the
 * absolute path it represents. Handles both POSIX and Windows separators
 * so users don't see broken navigation when scanning Windows datasets.
 */
function splitSegments(path: string): { label: string; absolute: string }[] {
  if (!path) return []
  const isWin = /^[A-Za-z]:[\\/]/.test(path) || path.includes("\\")
  const sep = isWin ? "\\" : "/"
  const parts = path.split(/[\\/]+/).filter(Boolean)

  const out: { label: string; absolute: string }[] = []
  let prefix = ""
  if (isWin && /^[A-Za-z]:$/.test(parts[0] ?? "")) {
    const root = `${parts[0]}\\`
    out.push({ label: parts[0], absolute: root })
    prefix = root
    parts.shift()
  } else if (path.startsWith("/")) {
    out.push({ label: "/", absolute: "/" })
    prefix = "/"
  } else {
    prefix = ""
  }
  for (const p of parts) {
    prefix = prefix ? joinSegment(prefix, p, sep) : p
    out.push({ label: p, absolute: prefix })
  }
  return out
}

function joinSegment(prefix: string, seg: string, sep: string): string {
  if (prefix.endsWith(sep)) return prefix + seg
  return prefix + sep + seg
}
