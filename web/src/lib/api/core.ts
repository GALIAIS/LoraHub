export const API_BASE = "/api"

// Default per-request timeout. Long enough that a remote LLM round-trip
// completes; short enough that a stuck server doesn't freeze the UI
// indefinitely. Overridable per-call via ``init.signal`` (the caller's
// signal takes precedence and can extend or shorten the deadline).
export const DEFAULT_TIMEOUT_MS = 30_000

/**
 * Structured error thrown by every API client function.
 *
 * Callers used to type-check on ``e instanceof Error`` and parse the
 * status out of ``e.message`` ("404 Not Found: …"). That's brittle —
 * a future refactor of the message shape would silently change the
 * branching. Prefer ``e instanceof ApiError && e.status === 404``;
 * the legacy message format is preserved for any straggler that still
 * uses ``startsWith("404 ")`` so existing call sites don't break.
 */
export class ApiError extends Error {
  readonly status: number
  /** Parsed JSON body when the response was JSON; raw text otherwise. */
  readonly body: unknown
  /** Path the request targeted (relative, sans API_BASE prefix). */
  readonly path: string

  constructor(status: number, statusText: string, body: unknown, path: string) {
    const detail =
      typeof body === "object" && body && "detail" in body
        ? (body as { detail: unknown }).detail
        : null
    const detailString =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" && "message" in detail
          ? String((detail as { message: unknown }).message)
          : typeof body === "string"
            ? body
            : ""
    super(`${status} ${statusText}${detailString ? `: ${detailString}` : ""}`)
    this.name = "ApiError"
    this.status = status
    this.body = body
    this.path = path
  }

  /** Structured preflight blockers, when the server returned them.
   *
   * The 422 response from `POST /api/jobs` (and resume / rerun) has the
   * shape `{detail: {message: "...", findings: [{...}]}}` when at least
   * one preflight category fired. Returns `null` when no findings list
   * is present so callers can fall through to plain `.message`.
   */
  get preflightFindings(): PreflightFinding[] | null {
    const body = this.body
    if (typeof body !== "object" || body === null) return null
    const detail = (body as { detail?: unknown }).detail
    if (typeof detail !== "object" || detail === null) return null
    const findings = (detail as { findings?: unknown }).findings
    if (!Array.isArray(findings)) return null
    return findings.filter(
      (f): f is PreflightFinding =>
        typeof f === "object" &&
        f !== null &&
        typeof (f as PreflightFinding).category === "string" &&
        typeof (f as PreflightFinding).field === "string" &&
        typeof (f as PreflightFinding).message === "string",
    )
  }

  /**
   * Structured import-error detail, when the server returned one. The
   * `/api/configs/import` endpoint emits `{type, message, line?, column?,
   * snippet?, hint?, kind?}` so the UI can render the offending line and
   * a Chinese-language remediation tip instead of the bare PyYAML
   * scanner string. Null when the body doesn't have a recognisable
   * detail object.
   */
  get importErrorDetail(): ImportErrorDetail | null {
    const body = this.body
    if (typeof body !== "object" || body === null) return null
    const detail = (body as { detail?: unknown }).detail
    if (typeof detail !== "object" || detail === null) return null
    const obj = detail as Record<string, unknown>
    // Distinguish a real import-error envelope from any other random
    // object body — the import endpoint always sets ``type``.
    if (typeof obj.type !== "string") return null
    return {
      type: obj.type,
      kind: typeof obj.kind === "string" ? obj.kind : undefined,
      message: typeof obj.message === "string" ? obj.message : undefined,
      line: typeof obj.line === "number" ? obj.line : undefined,
      column: typeof obj.column === "number" ? obj.column : undefined,
      snippet: typeof obj.snippet === "string" ? obj.snippet : undefined,
      hint: typeof obj.hint === "string" ? obj.hint : undefined,
    }
  }
}

/** Server-emitted detail for a failed `/api/configs/import` request. */
export interface ImportErrorDetail {
  type: string
  kind?: string
  message?: string
  line?: number
  column?: number
  snippet?: string
  hint?: string
}

/** A blocker / warning emitted by the API's preflight layer.
 *
 * Mirrors `lorahub.api.preflight.PreflightFinding.to_dict()`. Carry the
 * camelCase cfg path in `field` so the UI can highlight the input.
 */
export interface PreflightFinding {
  category: string
  severity: "info" | "warn" | "error"
  field: string
  message: string
  remediation: string
  extra?: Record<string, unknown>
}

export async function http<T>(path: string, init?: RequestInit): Promise<T> {
  // Wire in a default deadline. If the caller already passed a signal
  // we honour it; if not, ``AbortSignal.any`` lets the timeout race
  // alone. Either way the timeout fires after DEFAULT_TIMEOUT_MS unless
  // the caller's signal aborts first.
  const timeoutSignal =
    typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
      ? AbortSignal.timeout(DEFAULT_TIMEOUT_MS)
      : undefined
  const signals = [init?.signal, timeoutSignal].filter(
    (s): s is AbortSignal => Boolean(s),
  )
  const signal: AbortSignal | undefined =
    signals.length === 0
      ? undefined
      : signals.length === 1
        ? signals[0]
        : "any" in AbortSignal
          ? (AbortSignal as unknown as { any: (sigs: AbortSignal[]) => AbortSignal }).any(signals)
          : signals[0]

  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
    signal,
  })
  if (!res.ok) {
    const raw = await res.text()
    let body: unknown = raw
    try {
      body = JSON.parse(raw)
    } catch {
      // Plain-text error body (e.g. nginx 502 page) — keep as string.
    }
    throw new ApiError(res.status, res.statusText, body, path)
  }
  return res.json() as Promise<T>
}

/**
 * Iterate SSE frames from a fetch response body, yielding each parsed
 * payload. Handles CRLF / CR / LF terminators uniformly and skips
 * comment lines (``: ping``) and non-data fields.
 *
 * The ``terminalExec`` and ``applySystemUpdate`` paths used to inline
 * their own framing readers — three near-identical copies of the same
 * 30-line state machine. Bug fixes (CRLF normalisation, null body
 * guard) only landed in one of them. Funnel everything through here
 * so the next fix is a single edit.
 */
async function* iterSseFrames(
  res: Response,
): AsyncGenerator<string, void, void> {
  if (!res.body) {
    throw new Error("streaming response missing — does the proxy buffer SSE?")
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder
      .decode(value, { stream: true })
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
    let nl: number
    while ((nl = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, nl)
      buf = buf.slice(nl + 2)
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n")
      if (data) yield data
    }
  }
}

export async function readSseEvents<T>(
  res: Response,
  onEvent: (ev: T) => void,
): Promise<void> {
  for await (const frame of iterSseFrames(res)) {
    try {
      onEvent(JSON.parse(frame) as T)
    } catch {
      // Malformed frame — skip rather than tear down the whole stream.
    }
  }
}
