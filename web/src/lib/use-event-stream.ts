/**
 * Shared SSE-with-WebSocket-fallback hook.
 *
 * The workbench has three real-time channels:
 *   - per-job event stream (events array, cap ~500)
 *   - bootstrap session stream (events array, cap ~200)
 *   - system telemetry (single snapshot, replace-on-update)
 *
 * They used to live as three near-identical 80-line hooks in
 * `lib/api.ts`. The duplication kept getting copy-paste edited
 * inconsistently (e.g. tab-visibility reconnect was added to
 * system telemetry only). This hook keeps the connection plumbing
 * in one place and lets each consumer say what it wants done with
 * each frame via `reduce(state, parsed) => state`.
 *
 * Each consumer wraps this in a tiny adapter (see
 * `useJobStream` / `useBootstrapStream` / `useSystemStream` in
 * `lib/api.ts`) so call sites stay readable.
 */
import { useEffect, useRef, useState } from "react"

export type StreamStatus = "idle" | "open" | "closed"

export interface UseEventStreamOptions<TState, TPayload> {
  /**
   * The SSE endpoint, e.g. `/api/jobs/abc/sse`. Pass `null` to
   * disable the hook (no connection is opened, no listeners
   * mount). When this changes the previous connection tears down
   * and a fresh one opens.
   */
  ssePath: string | null
  /**
   * The legacy WS endpoint, e.g. `/api/jobs/abc/stream`. Used only
   * when the runtime lacks `EventSource`. Same nullable semantics
   * as `ssePath`.
   */
  wsPath: string | null
  /** Initial state; also re-applied on (re)connect. */
  initialState: TState
  /**
   * Reducer applied to every parsed JSON frame. Pure — return a
   * fresh state value instead of mutating.
   */
  reduce: (prev: TState, parsed: TPayload) => TState
  /**
   * Optional pre-filter: return `true` to drop the frame before it
   * reaches the reducer. Used by job/bootstrap streams to swallow
   * `{type:"ping"}` keepalive frames the legacy WS endpoint emits.
   */
  shouldDrop?: (parsed: TPayload) => boolean
  /**
   * Whether the WS branch should reconnect aggressively on tab
   * focus / network online. The system telemetry hook needs this;
   * the per-job stream doesn't (the SSE branch is the hot path).
   */
  reconnectOnVisibility?: boolean
}

export interface UseEventStreamResult<TState> {
  state: TState
  status: StreamStatus
}

/**
 * Connection lifecycle (SSE branch):
 *   - The browser auto-reconnects EventSource after `retry: <ms>`,
 *     so we just toggle status on `onerror`. The server emits
 *     `id: N` on every frame and the browser sends it back as
 *     `Last-Event-ID` on reconnect, so the resume layer is the
 *     server's job.
 *   - On unmount we `.close()` to stop reconnect attempts.
 *
 * Connection lifecycle (WS branch):
 *   - Exponential backoff up to 3-5 s, mirroring the prior hand-
 *     written hooks.
 *   - When `reconnectOnVisibility` is on, we also reconnect on
 *     `visibilitychange:visible` and `window:online` because long-
 *     lived WS connections silently die after laptop sleep.
 */
export function useEventStream<TState, TPayload = unknown>(
  options: UseEventStreamOptions<TState, TPayload>,
): UseEventStreamResult<TState> {
  const {
    ssePath,
    wsPath,
    initialState,
    reduce,
    shouldDrop,
    reconnectOnVisibility = false,
  } = options

  const [state, setState] = useState<TState>(initialState)
  const [status, setStatus] = useState<StreamStatus>("idle")
  // Hold the live source so unmount can close it. Effect deps decide
  // when we tear down — never read this ref to derive UI state.
  const sourceRef = useRef<EventSource | WebSocket | null>(null)

  // Bind reducer / drop / initial-state by ref so the connection
  // effect doesn't re-fire when callers pass inline closures. The
  // effect only re-runs when the URL or visibility flag changes.
  const reduceRef = useRef(reduce)
  const dropRef = useRef(shouldDrop)
  const initialRef = useRef(initialState)
  reduceRef.current = reduce
  dropRef.current = shouldDrop
  initialRef.current = initialState

  useEffect(() => {
    if (!ssePath && !wsPath) return
    let cancelled = false
    const useSse = !!ssePath && typeof EventSource !== "undefined"

    // Reset to the initial state on every fresh subscription so a
    // job-id swap doesn't bleed events from the previous job.
    setState(initialRef.current)
    setStatus("idle")

    function applyFrame(parsed: TPayload) {
      if (dropRef.current?.(parsed)) return
      setState((prev) => reduceRef.current(prev, parsed))
    }

    function safeParse(text: string): TPayload | undefined {
      try {
        return JSON.parse(text) as TPayload
      } catch {
        return undefined
      }
    }

    function connectSse(path: string) {
      const es = new EventSource(path)
      sourceRef.current = es
      es.onopen = () => setStatus("open")
      es.onerror = () => setStatus("closed")
      es.onmessage = (msg) => {
        const parsed = safeParse(msg.data)
        if (parsed !== undefined) applyFrame(parsed)
      }
    }

    function connectWs(path: string) {
      let ws: WebSocket | null = null
      let backoff = 0
      let retryTimer: ReturnType<typeof setTimeout> | null = null

      function open() {
        if (cancelled) return
        if (typeof navigator !== "undefined" && !navigator.onLine) return
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
        const host = window.location.host || "127.0.0.1:18765"
        ws = new WebSocket(`${protocol}//${host}${path}`)
        sourceRef.current = ws
        ws.onopen = () => {
          backoff = 0
          setStatus("open")
        }
        ws.onclose = () => {
          setStatus("closed")
          if (cancelled) return
          retryTimer = setTimeout(open, backoff)
          backoff = backoff === 0 ? 500 : Math.min(backoff * 2, 5000)
        }
        ws.onerror = () => {}
        ws.onmessage = (msg) => {
          const parsed = safeParse(msg.data)
          if (parsed !== undefined) applyFrame(parsed)
        }
      }

      function reconnectNow() {
        if (cancelled) return
        if (ws && ws.readyState === WebSocket.OPEN) return
        if (retryTimer !== null) clearTimeout(retryTimer)
        retryTimer = null
        backoff = 0
        open()
      }

      function onVis() {
        if (document.visibilityState === "visible") reconnectNow()
      }

      if (reconnectOnVisibility) {
        document.addEventListener("visibilitychange", onVis)
        window.addEventListener("online", reconnectNow)
        // Slight stagger so `cancelled` has a chance to flip if the
        // effect's cleanup runs synchronously.
        retryTimer = setTimeout(open, 30)
      } else {
        open()
      }

      return () => {
        if (reconnectOnVisibility) {
          document.removeEventListener("visibilitychange", onVis)
          window.removeEventListener("online", reconnectNow)
        }
        if (retryTimer !== null) clearTimeout(retryTimer)
      }
    }

    let cleanupWs: (() => void) | undefined
    if (useSse && ssePath) connectSse(ssePath)
    else if (wsPath) cleanupWs = connectWs(wsPath)

    return () => {
      cancelled = true
      cleanupWs?.()
      const s = sourceRef.current
      if (s) {
        if (s instanceof WebSocket) {
          // Detach handlers first so close events don't trigger a
          // reconnect schedule against a tearing-down hook.
          s.onclose = null
          s.onerror = null
          if (s.readyState === WebSocket.OPEN) s.close()
        } else {
          s.close()
        }
        sourceRef.current = null
      }
    }
  }, [ssePath, wsPath, reconnectOnVisibility])

  return { state, status }
}
