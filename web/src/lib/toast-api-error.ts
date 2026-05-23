/**
 * Render an API failure as a sonner toast.
 *
 * For API responses that carry a `findings` list (the preflight layer
 * returns these on `POST /api/jobs` 422), we expand each finding into a
 * dedicated row underneath the headline so the user can read every
 * blocker without cracking open devtools. For everything else we fall
 * back to the plain error message — same as a `toast.error(label, {
 * description: err.message })` would have done.
 *
 * The same ApiError also flows into the persistent error registry via
 * ``reportError`` so users can find a record of the failure later from
 * Settings → 错误上报 — preflight 422s and silent toast dismissals both
 * leave a trail behind.
 */
import { toast } from "sonner"
import { ApiError, type PreflightFinding } from "@/lib/api"
import { reportError } from "@/lib/error-reporter"

export interface ToastApiErrorOptions {
  /** Headline shown as the toast title. */
  title: string
}

export function toastApiError(
  error: unknown,
  options: ToastApiErrorOptions,
): void {
  const { title } = options
  if (error instanceof ApiError) {
    const findings = error.preflightFindings
    if (findings && findings.length > 0) {
      toast.error(title, {
        description: renderFindingsDescription(findings),
        duration: 14_000,
      })
      void reportError({
        severity: "warn",
        source: "frontend.api",
        category: "preflight",
        title,
        message: error.message,
        context: {
          findings,
          status: error.status,
          path: error.path,
        },
        requestPath: error.path,
      })
      return
    }
  }
  toast.error(title, {
    description: error instanceof Error ? error.message : String(error),
  })
  // Persist non-preflight errors too so the registry covers every UI
  // failure path. We tag the API status when available — that's enough
  // for the registry's filter chips to differentiate 4xx from 5xx.
  void reportError({
    severity:
      error instanceof ApiError && error.status >= 500 ? "error" : "warn",
    source: "frontend.api",
    category:
      error instanceof ApiError ? `http_${error.status}` : "uncaught",
    title,
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack ?? null : null,
    context:
      error instanceof ApiError
        ? { status: error.status, path: error.path, body: error.body }
        : {},
    requestPath: error instanceof ApiError ? error.path : null,
  })
}

function renderFindingsDescription(findings: PreflightFinding[]): string {
  // sonner accepts a string OR a ReactNode; we pick string to keep this
  // helper framework-light. Each line carries the field + message;
  // remediation goes on a follow-up indented line so the text stays
  // scannable in the small toast viewport.
  return findings
    .map((f) => {
      const head = `[${f.severity}] ${f.field}: ${f.message}`
      return f.remediation ? `${head}\n  → ${f.remediation}` : head
    })
    .join("\n")
}
