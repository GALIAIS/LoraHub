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
 * Backend-captured failures carry a report id and are not submitted again.
 * Expected 4xx validation responses remain toasts rather than error records.
 */
import { toast } from "sonner"
import { ApiError, type PreflightFinding } from "@/lib/api"
import { reportError } from "@/lib/error-reporter"
import { fieldLabelFor, prettifyFieldPath } from "@/lib/field-labels"

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
      return
    }
  }
  toast.error(title, {
    description: error instanceof Error ? error.message : String(error),
  })
  if (
    error instanceof ApiError &&
    (error.backendReportId !== null || error.status < 500)
  ) {
    return
  }
  // Persist transport/proxy failures that the backend could not capture.
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
    requestId: error instanceof ApiError ? error.requestId : null,
    requestPath: error instanceof ApiError ? error.path : null,
  })
}

function renderFindingsDescription(findings: PreflightFinding[]): string {
  // sonner accepts a string OR a ReactNode; we pick string to keep this
  // helper framework-light. Each line carries the friendly Chinese
  // label first (so first-time users can find the affected control),
  // then the raw dotted path in parentheses for power users, then the
  // message; remediation goes on a follow-up indented line so the
  // text stays scannable in the small toast viewport.
  return findings
    .map((f) => {
      const friendly = fieldLabelFor(f.field) ?? prettifyFieldPath(f.field)
      const showRaw = friendly !== f.field
      const head = showRaw
        ? `[${f.severity}] ${friendly} (${f.field}): ${f.message}`
        : `[${f.severity}] ${friendly}: ${f.message}`
      return f.remediation ? `${head}\n  → ${f.remediation}` : head
    })
    .join("\n")
}
