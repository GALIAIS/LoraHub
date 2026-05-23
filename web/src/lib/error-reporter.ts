/**
 * Client-side error reporter.
 *
 * Funnels every interesting frontend failure into the backend's
 * ``POST /api/error-reports`` endpoint so the user can review the
 * registry from Settings → 错误上报. Mirrors the shape of
 * ``lorahub.api.error_reporter.capture`` so server-side and
 * client-side reports share categories.
 *
 * Design notes:
 * - Fire-and-forget: a failing report POST must never throw, otherwise
 *   the original error message could be drowned by recursive logging.
 * - Lightweight in-memory dedup: the same `${source}:${title}:${message}`
 *   triplet within ``DEDUP_WINDOW_MS`` is dropped so a render loop
 *   doesn't flood SQLite (which the backend already bounds at 5 000
 *   rows but we'd rather not spend the round-trips).
 * - The POST goes through ``fetch`` directly with ``keepalive`` so it
 *   survives a navigation away while the user is leaving the broken
 *   page.
 */

export type ReportSeverity = "info" | "warn" | "error" | "fatal"

export interface ReportPayload {
  severity?: ReportSeverity
  source: string
  category: string
  title: string
  message: string
  stack?: string | null
  context?: Record<string, unknown>
  jobId?: string | null
  requestId?: string | null
  requestPath?: string | null
}

const ENDPOINT = "/api/error-reports"
const DEDUP_WINDOW_MS = 5_000

const recent = new Map<string, number>()

function dedupKey(p: ReportPayload): string {
  // Truncate so a 50 KB stack difference doesn't defeat the cache —
  // we want all variants of the same render-loop crash collapsed.
  const msg = (p.message || "").slice(0, 200)
  return `${p.source}:${p.title}:${msg}`
}

function shouldSend(p: ReportPayload): boolean {
  const key = dedupKey(p)
  const last = recent.get(key)
  const now = Date.now()
  if (last !== undefined && now - last < DEDUP_WINDOW_MS) {
    return false
  }
  recent.set(key, now)
  // Bound the cache so a long-lived tab doesn't leak.
  if (recent.size > 200) {
    const cutoff = now - DEDUP_WINDOW_MS
    for (const [k, ts] of recent) {
      if (ts < cutoff) recent.delete(k)
    }
  }
  return true
}

let enabled = true

export function setReportingEnabled(value: boolean): void {
  enabled = value
  try {
    localStorage.setItem("lorahub.error-reporting.enabled", value ? "1" : "0")
  } catch {
    /* private mode / quota — non-fatal */
  }
}

export function getReportingEnabled(): boolean {
  try {
    const raw = localStorage.getItem("lorahub.error-reporting.enabled")
    if (raw === "0") return false
  } catch {
    /* ignore */
  }
  return enabled
}

export async function reportError(p: ReportPayload): Promise<string | null> {
  if (!getReportingEnabled()) return null
  if (!shouldSend(p)) return null
  const body = {
    severity: p.severity ?? "error",
    source: p.source,
    category: p.category,
    title: p.title.slice(0, 300),
    message: p.message.slice(0, 20_000),
    stack: p.stack ? p.stack.slice(0, 200_000) : null,
    context: p.context ?? {},
    job_id: p.jobId ?? null,
    request_id: p.requestId ?? null,
    request_path: p.requestPath ?? null,
  }
  try {
    const resp = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // Survive a tab close so a crash on unload still records.
      keepalive: true,
    })
    if (!resp.ok) return null
    const json = (await resp.json()) as { id?: string }
    return json.id ?? null
  } catch {
    // Reporter must never throw — losing one row is better than
    // pulling the whole UI down with a recursive failure.
    return null
  }
}
