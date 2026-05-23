/**
 * Analysis page view mode.
 *
 * The same analysis workbench serves three audiences with very
 * different mental models:
 *
 *   - Live (running job): the user wants to know "is this run on
 *     track?" Three questions dominate: convergence, overfit,
 *     finishing on time. Detail panels (metric grid, checkpoint
 *     playback) add cognitive load without the corresponding
 *     reward, so we collapse them by default.
 *
 *   - Postmortem (terminal job): the user is debugging or doing
 *     research. Every signal is fair game; everything expanded.
 *
 *   - Custom: the moment the user toggles a panel manually we stop
 *     applying mode defaults so their explicit choice persists for
 *     the rest of the session.
 *
 * Default mode is auto-derived from the job state (live vs
 * terminal). The user can manually switch in the workbench header;
 * preference persists per-job via sessionStorage so navigating away
 * and back keeps the same view.
 */

const LS_KEY_PREFIX = "lorahub.analysis.viewMode."
const PANEL_PREFIX = "lorahub.analysis.panel."

export type ViewMode = "live" | "postmortem" | "custom"

export interface PanelState {
  showStageTimeline: boolean
  showMetricGrid: boolean
  showCheckpointPlayback: boolean
}

const LIVE_DEFAULTS: PanelState = {
  showStageTimeline: true,
  showMetricGrid: false,
  showCheckpointPlayback: false,
}

const POSTMORTEM_DEFAULTS: PanelState = {
  showStageTimeline: true,
  showMetricGrid: true,
  showCheckpointPlayback: true,
}

export function defaultPanels(mode: ViewMode): PanelState {
  if (mode === "live") return LIVE_DEFAULTS
  if (mode === "postmortem") return POSTMORTEM_DEFAULTS
  // Custom mode falls back to postmortem until the user mutates a panel.
  return POSTMORTEM_DEFAULTS
}

export function loadViewMode(jobId: string): ViewMode | null {
  if (typeof window === "undefined") return null
  try {
    const raw = window.sessionStorage.getItem(LS_KEY_PREFIX + jobId)
    if (raw === "live" || raw === "postmortem" || raw === "custom") return raw
  } catch {
    // ignore corrupt storage
  }
  return null
}

export function saveViewMode(jobId: string, mode: ViewMode): void {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(LS_KEY_PREFIX + jobId, mode)
  } catch {
    // quota-exceeded; not fatal
  }
}

export function loadPanelOverrides(jobId: string): Partial<PanelState> | null {
  if (typeof window === "undefined") return null
  try {
    const raw = window.sessionStorage.getItem(PANEL_PREFIX + jobId)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<PanelState>
    return parsed
  } catch {
    return null
  }
}

export function savePanelOverrides(
  jobId: string,
  overrides: Partial<PanelState>,
): void {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(
      PANEL_PREFIX + jobId,
      JSON.stringify(overrides),
    )
  } catch {
    // quota-exceeded; not fatal
  }
}

export function inferDefaultMode(isTerminal: boolean): ViewMode {
  return isTerminal ? "postmortem" : "live"
}
