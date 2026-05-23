/**
 * Helpers for switching the analysis charts' X-axis between training
 * step, epoch and wall-clock seconds. Each backend logs (step, epoch?,
 * timestamp) per metric event, so the conversion is purely a mapping
 * function — no extra fetches required.
 */

import type { JobMetricsResponse } from "@/lib/api"

export type XMode = "step" | "epoch" | "wallclock"

const SS_KEY_PREFIX = "lorahub.analysis.xMode."

export function loadXMode(jobId: string): XMode {
  if (typeof window === "undefined") return "step"
  try {
    const raw = window.sessionStorage.getItem(SS_KEY_PREFIX + jobId)
    if (raw === "step" || raw === "epoch" || raw === "wallclock") return raw
  } catch {
    // ignore corrupt storage
  }
  return "step"
}

export function saveXMode(jobId: string, mode: XMode): void {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(SS_KEY_PREFIX + jobId, mode)
  } catch {
    // quota-exceeded; not fatal
  }
}

export function xModeLabel(mode: XMode): string {
  switch (mode) {
    case "step":
      return "step"
    case "epoch":
      return "epoch"
    case "wallclock":
      return "wallclock(s)"
  }
}

/**
 * Format an X-axis tick value according to mode. Wallclock seconds get
 * a compact ``H:MM:SS`` rendering when ≥ 60 s, otherwise raw seconds.
 */
export function formatXTick(mode: XMode): (v: number) => string {
  if (mode === "wallclock") {
    return (v: number) => {
      if (!Number.isFinite(v) || v < 0) return "—"
      if (v < 60) return `${v.toFixed(0)}s`
      const h = Math.floor(v / 3600)
      const m = Math.floor((v % 3600) / 60)
      const s = Math.floor(v % 60)
      if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
      return `${m}:${String(s).padStart(2, "0")}`
    }
  }
  return (v: number) => {
    if (!Number.isFinite(v)) return "—"
    return Math.round(v).toString()
  }
}

/**
 * Build a function that maps a (step, epoch?, ts?) tuple onto the
 * configured X-axis value. ``ts`` is taken relative to the first
 * step's timestamp so wallclock starts at zero. When the requested
 * mode has missing data on a sample, we fall back to step so the
 * chart never shows a partial line; calling code can detect that
 * by passing ``strict=true`` to drop unmappable points.
 */
export function xMapper(
  mode: XMode,
  metrics: JobMetricsResponse | null,
): (sample: { step: number; epoch?: number | null; ts?: number }) => number {
  if (mode === "step") return (s) => s.step
  if (mode === "epoch") {
    return (s) => {
      if (typeof s.epoch === "number" && Number.isFinite(s.epoch)) return s.epoch
      return s.step
    }
  }
  // wallclock — treat the first step's ts as t=0 so the chart's X
  // origin lines up with "training started", not Unix epoch.
  const t0 = metrics?.first_step_ts ?? null
  return (s) => {
    if (t0 != null && typeof s.ts === "number" && Number.isFinite(s.ts)) {
      return Math.max(0, s.ts - t0)
    }
    return s.step
  }
}
