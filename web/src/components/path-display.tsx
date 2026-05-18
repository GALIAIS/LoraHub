/**
 * PathDisplay — render a long filesystem path with two tweaks the
 * built-in `truncate` + `title` combo doesn't get right:
 *
 *   1. Tail-preserving truncation: workspace paths look like
 *      `/very/long/.../runs/jobs/abc123` and the *interesting* part
 *      is the right-hand tail (the run id). A plain `truncate` cuts
 *      from the right and hides exactly what the user wants. We
 *      render the last N segments as the visible body and fold the
 *      rest into a leading ellipsis.
 *
 *   2. Touch-friendly tooltip: `title` only fires on mouse hover.
 *      The base-ui `<Tooltip>` we already use elsewhere works on
 *      keyboard focus + long-press too, and renders inside the
 *      design system rather than the browser's beige rectangle.
 *
 * Both behaviours degrade gracefully — when the path is short
 * enough, the leading ellipsis is dropped and the tooltip stays
 * silent.
 */
import { useMemo } from "react"
import { cn } from "@/lib/utils"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface PathDisplayProps {
  path: string
  /**
   * How many trailing segments to keep visible. 2 means "last two
   * directory levels" (e.g. `runs/lx_anima_style_lora`).
   */
  tailSegments?: number
  /**
   * Override the threshold below which the path is rendered intact
   * with no ellipsis prefix. Defaults to 32 characters which fits
   * the typical sidebar / chip width.
   */
  inlineMaxChars?: number
  /**
   * Render as block (true) or inline-flex (false). Block lets the
   * caller wrap the component in a flex container without it
   * collapsing.
   */
  block?: boolean
  className?: string
}

const SEP = /[\\/]/

export function PathDisplay({
  path,
  tailSegments = 2,
  inlineMaxChars = 32,
  block = false,
  className,
}: PathDisplayProps) {
  const display = useMemo(() => formatPath(path, tailSegments, inlineMaxChars), [
    path,
    tailSegments,
    inlineMaxChars,
  ])

  // Short paths don't need a tooltip — the visible text is already
  // the full thing. Skip the wrapper to avoid taking a tab stop /
  // hover handler the user doesn't need.
  if (!display.truncated) {
    return (
      <span
        className={cn(
          "font-mono text-[12px] text-muted-foreground truncate",
          block ? "block" : "inline-flex",
          className,
        )}
      >
        {display.text}
      </span>
    )
  }

  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger
          className={cn(
            "font-mono text-[12px] text-muted-foreground truncate cursor-help bg-transparent border-0 p-0 text-left",
            block ? "block w-full" : "inline-flex",
            className,
          )}
          // Keep `title` as a fallback for environments that don't
          // mount portals (server-rendered html, no-JS test runs).
          title={path}
        >
          {display.text}
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          className="max-w-[min(80vw,640px)] break-all font-mono text-[11px]"
        >
          {path}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

interface FormattedPath {
  text: string
  truncated: boolean
}

function formatPath(
  raw: string,
  tail: number,
  inlineMax: number,
): FormattedPath {
  const path = raw.trim()
  if (!path) return { text: "", truncated: false }
  if (path.length <= inlineMax) return { text: path, truncated: false }
  const segments = path.split(SEP).filter(Boolean)
  if (segments.length <= tail + 1) {
    // Even the original couldn't be split usefully; fall back to
    // the full path with an inline-truncate visual.
    return { text: path, truncated: true }
  }
  // Pick the trailing N segments. Preserve the original separator
  // between them — Linux paths use `/`, Windows `\`. We can't tell
  // from a stripped segment list, so re-join with `/` (the more
  // forgiving choice; Windows users still parse it correctly).
  const tailSegments = segments.slice(-tail).join("/")
  return { text: `…/${tailSegments}`, truncated: true }
}
