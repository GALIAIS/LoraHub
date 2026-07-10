export const API_BASE = "/api"

// Default per-request timeout. Long enough that a remote LLM round-trip
// completes; short enough that a stuck server doesn't freeze the UI
// indefinitely. Overridable per-call via ``init.signal`` (the caller's
// signal takes precedence and can extend or shorten the deadline).
export const DEFAULT_TIMEOUT_MS = 30_000
const MAX_SSE_BUFFER_CHARS = 2 * 1024 * 1024

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
  /** Correlation id returned by the API or reverse proxy. */
  readonly requestId: string | null

  constructor(
    status: number,
    statusText: string,
    body: unknown,
    path: string,
    requestId: string | null = null,
  ) {
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
    this.requestId = requestId ?? this.bodyRequestId
  }

  get backendReportId(): string | null {
    const detail = this.detailObject
    return typeof detail?.report_id === "string" ? detail.report_id : null
  }

  private get bodyRequestId(): string | null {
    const detail = this.detailObject
    return typeof detail?.request_id === "string" ? detail.request_id : null
  }

  private get detailObject(): Record<string, unknown> | null {
    if (typeof this.body !== "object" || this.body === null) return null
    const detail = (this.body as { detail?: unknown }).detail
    return typeof detail === "object" && detail !== null
      ? (detail as Record<string, unknown>)
      : null
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
  // An explicit caller signal owns the deadline. This is how long-running
  // operations opt into a larger timeout without racing the 30-second default.
  const signal = init?.signal ?? timeoutSignal

  const headers = new Headers(init?.headers)
  const bodyIsFormData =
    typeof FormData !== "undefined" && init?.body instanceof FormData
  if (init?.body != null && !bodyIsFormData && !headers.has("content-type")) {
    headers.set("content-type", "application/json")
  }
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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
    throw new ApiError(
      res.status,
      res.statusText,
      body,
      path,
      res.headers.get("x-request-id"),
    )
  }
  if (res.status === 204 || res.status === 205) return undefined as T
  const raw = await res.text()
  if (!raw) return undefined as T
  const contentType = res.headers.get("content-type")?.toLowerCase() ?? ""
  if (contentType.includes("json")) return JSON.parse(raw) as T
  return raw as unknown as T
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
  let pendingCarriageReturn = false
  let completed = false

  function appendDecoded(text: string, final = false): void {
    if (pendingCarriageReturn) {
      text = `\r${text}`
      pendingCarriageReturn = false
    }
    if (!final && text.endsWith("\r")) {
      pendingCarriageReturn = true
      text = text.slice(0, -1)
    }
    buf += text.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
    if (buf.length > MAX_SSE_BUFFER_CHARS) {
      throw new Error("SSE frame exceeded the client buffer limit")
    }
  }

  function takeFrames(final = false): string[] {
    const frames: string[] = []
    let nl: number
    while ((nl = buf.indexOf("\n\n")) >= 0) {
      frames.push(buf.slice(0, nl))
      buf = buf.slice(nl + 2)
    }
    if (final && buf.trim()) {
      frames.push(buf)
      buf = ""
    }
    return frames
  }

  function frameData(frame: string): string {
    return frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n")
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        completed = true
        break
      }
      appendDecoded(decoder.decode(value, { stream: true }))
      for (const frame of takeFrames()) {
        const data = frameData(frame)
        if (data) yield data
      }
    }
    appendDecoded(decoder.decode(), true)
    for (const frame of takeFrames(true)) {
      const data = frameData(frame)
      if (data) yield data
    }
  } finally {
    if (!completed) {
      await reader.cancel().catch(() => undefined)
    }
    reader.releaseLock()
  }
}

export async function readSseEvents<T>(
  res: Response,
  onEvent: (ev: T) => void,
): Promise<void> {
  for await (const frame of iterSseFrames(res)) {
    let parsed: T
    try {
      parsed = JSON.parse(frame) as T
    } catch {
      // Malformed frame — skip rather than tear down the whole stream.
      continue
    }
    // Callback failures are application errors and must propagate to the
    // caller; swallowing them here makes an update or terminal stream look
    // successful while the UI silently stops processing events.
    onEvent(parsed)
  }
}
