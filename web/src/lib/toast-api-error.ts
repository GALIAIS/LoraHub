/**
 * Render an API failure as a sonner toast.
 *
 * For API responses that carry a `findings` list (the preflight layer
 * returns these on `POST /api/jobs` 422), we expand each finding into a
 * dedicated row underneath the headline so the user can read every
 * blocker without cracking open devtools. For everything else we fall
 * back to the plain error message — same as a `toast.error(label, {
 * description: err.message })` would have done.
 */
import { toast } from "sonner"
import { ApiError, type PreflightFinding } from "@/lib/api"

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
