import { http, ApiError, readSseEvents, API_BASE } from "./core"

// --------------------------------------------------------------------------- //
// Terminal — venv-scoped command runner
// --------------------------------------------------------------------------- //

export interface TerminalEnvironment {
  backend_id: string
  name: string
  repo_path: string
  python_path: string | null
  venv_dir: string | null
  venv_detected: boolean
  ready: boolean
  prompt: string
}

export interface TerminalSessionsResponse {
  backends: TerminalEnvironment[]
  default_backend: string
  unrestricted: boolean
  command_timeout_s: number
}

export type TerminalEvent =
  | { type: "start"; argv: string[]; cwd: string }
  | { type: "stdout"; data: string }
  | { type: "stderr"; data: string }
  | { type: "exit"; code: number }
  | { type: "error"; data: string }

export async function terminalListSessions(): Promise<TerminalSessionsResponse> {
  return http<TerminalSessionsResponse>("/terminal/sessions")
}

/**
 * Stream a terminal command from the backend, calling ``onEvent`` for
 * each SSE frame as it arrives. Returns an async function that resolves
 * once the stream completes (either via an ``exit`` event or the abort
 * signal). The caller passes an AbortController to cancel mid-stream.
 *
 * Why fetch + ReadableStream and not EventSource: EventSource doesn't
 * let us POST a request body, and we need to send {backend_id, command}
 * via JSON. Manual SSE parsing is short and gives us abort + headers.
 */
export async function terminalExec(
  body: { backend_id: string; command: string },
  opts: {
    signal?: AbortSignal
    onEvent: (event: TerminalEvent) => void
  },
): Promise<void> {
  const res = await fetch(`${API_BASE}/terminal/exec`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal: opts.signal,
  })
  if (!res.ok) {
    let errBody: unknown = ""
    try {
      errBody = await res.json()
    } catch {
      try {
        errBody = await res.text()
      } catch {
        /* empty */
      }
    }
    throw new ApiError(res.status, res.statusText, errBody, "/terminal/exec")
  }
  await readSseEvents<TerminalEvent>(res, opts.onEvent)
}
