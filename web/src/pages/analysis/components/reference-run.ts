/**
 * Reference run — a globally pinned historical job that other analysis
 * pages overlay against their own loss curve. Persisted in
 * localStorage so it survives reloads, scoped to the entire installation
 * (not per-job). One slot at a time keeps the UI legible.
 */

const LS_KEY = "lorahub.analysis.referenceRun"

export interface ReferenceRun {
  jobId: string
  label: string
}

export function loadReferenceRun(): ReferenceRun | null {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(LS_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as ReferenceRun
    if (
      parsed &&
      typeof parsed.jobId === "string" &&
      typeof parsed.label === "string"
    )
      return parsed
  } catch {
    // ignore
  }
  return null
}

export function saveReferenceRun(ref: ReferenceRun | null): void {
  if (typeof window === "undefined") return
  try {
    if (ref) {
      window.localStorage.setItem(LS_KEY, JSON.stringify(ref))
    } else {
      window.localStorage.removeItem(LS_KEY)
    }
    window.dispatchEvent(new CustomEvent("lorahub:reference-run-changed"))
  } catch {
    // quota-exceeded; not fatal
  }
}
