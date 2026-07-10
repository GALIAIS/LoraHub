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
   * Whether the WS branch should reconnect aggressively on tab focus.
   * Network-online recovery is always enabled for both transports.
   */
  reconnectOnVisibility?: boolean
}

export interface UseEventStreamResult<TState> {
  state: TState
  status: StreamStatus
}

/**
 * Connection lifecycle (SSE branch):
 *   - The browser auto-reconnects EventSource when its ``readyState``
 *     is ``CONNECTING``. After certain server failures (5xx, the
 *     server explicitly closing the response stream) the
 *     ``readyState`` flips to ``CLOSED`` and the browser stops
 *     trying — silent permanent disconnect.
 *   - We watch for that transition in ``onerror``: if the source
 *     entered ``CLOSED`` we tear it down and schedule our own
 *     reconnect with exponential backoff (500ms → 8s, capped).
 *   - The server emits ``id: N`` on every frame; reconnects pass
 *     it back as ``Last-Event-ID`` automatically so resume is
 *     handled server-side.
 *   - Tab visibility / online events shortcut the backoff so the
 *     UI reconnects instantly when the user comes back to it.
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
    if (!ssePath && !wsPath) {
      setState(initialRef.current)
      setStatus("idle")
      return
    }
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

    function connectSse(path: string): () => void {
      let es: EventSource | null = null
      let backoff = 0
      let retryTimer: ReturnType<typeof setTimeout> | null = null
      let lastEventId = ""

      function resumePath(): string {
        if (!lastEventId) return path
        const url = new URL(path, window.location.href)
        url.searchParams.set("lastEventId", lastEventId)
        return url.toString()
      }

      function open() {
        if (cancelled) return
        const source = new EventSource(resumePath())
        es = source
        sourceRef.current = source
        source.onopen = () => {
          if (cancelled || es !== source) return
          backoff = 0
          setStatus("open")
        }
        source.onmessage = (msg) => {
          if (cancelled || es !== source) return
          if (msg.lastEventId) lastEventId = msg.lastEventId
          const parsed = safeParse(msg.data)
          if (parsed !== undefined) applyFrame(parsed)
        }
        source.onerror = () => {
          if (cancelled || es !== source) return
          // ``readyState`` distinguishes:
          //   CONNECTING (0) — browser is already retrying for us
          //   OPEN       (1) — transient blip, browser handles it
          //   CLOSED     (2) — browser gave up; permanent without
          //                    our intervention.
          // We only step in for the CLOSED case so well-behaved
          // hiccups don't double-reconnect.
          setStatus("closed")
          if (source.readyState !== EventSource.CLOSED) return
          source.onopen = null
          source.onmessage = null
          source.onerror = null
          source.close()
          es = null
          if (sourceRef.current === source) sourceRef.current = null
          if (retryTimer !== null) clearTimeout(retryTimer)
          retryTimer = setTimeout(open, backoff)
          backoff = backoff === 0 ? 500 : Math.min(backoff * 2, 8000)
        }
      }

      function reconnectNow() {
        if (cancelled) return
        // If the connection is already healthy, don't churn it.
        if (es && es.readyState === EventSource.OPEN) return
        if (retryTimer !== null) clearTimeout(retryTimer)
        retryTimer = null
        if (es) {
          es.close()
          es = null
          sourceRef.current = null
        }
        backoff = 0
        open()
      }

      function onVis() {
        if (document.visibilityState === "visible") reconnectNow()
      }

      // SSE always wants visibility / online recovery — the browser's
      // built-in reconnect doesn't reach here for the CLOSED state,
      // and a sleeping laptop loses the underlying TCP socket either
      // way. Cheap to wire up unconditionally.
      document.addEventListener("visibilitychange", onVis)
      window.addEventListener("online", reconnectNow)

      open()

      return () => {
        document.removeEventListener("visibilitychange", onVis)
        window.removeEventListener("online", reconnectNow)
        if (retryTimer !== null) clearTimeout(retryTimer)
      }
    }

    function connectWs(path: string) {
      let ws: WebSocket | null = null
      let backoff = 0
      let retryTimer: ReturnType<typeof setTimeout> | null = null

      function open() {
        if (cancelled) return
        if (typeof navigator !== "undefined" && !navigator.onLine) return
        if (
          ws &&
          (ws.readyState === WebSocket.OPEN ||
            ws.readyState === WebSocket.CONNECTING)
        ) {
          return
        }
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
        const host = window.location.host || "127.0.0.1:18765"
        const socket = new WebSocket(`${protocol}//${host}${path}`)
        ws = socket
        sourceRef.current = socket
        socket.onopen = () => {
          if (cancelled || ws !== socket) return
          backoff = 0
          setStatus("open")
        }
        socket.onclose = () => {
          if (cancelled || ws !== socket) return
          ws = null
          if (sourceRef.current === socket) sourceRef.current = null
          setStatus("closed")
          if (retryTimer !== null) clearTimeout(retryTimer)
          retryTimer = setTimeout(open, backoff)
          backoff = backoff === 0 ? 500 : Math.min(backoff * 2, 5000)
        }
        socket.onerror = () => {}
        socket.onmessage = (msg) => {
          if (cancelled || ws !== socket) return
          const parsed = safeParse(msg.data)
          if (parsed !== undefined) applyFrame(parsed)
        }
      }

      function reconnectNow() {
        if (cancelled) return
        if (
          ws &&
          (ws.readyState === WebSocket.OPEN ||
            ws.readyState === WebSocket.CONNECTING)
        ) {
          return
        }
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
        // Slight stagger so `cancelled` has a chance to flip if the
        // effect's cleanup runs synchronously.
        retryTimer = setTimeout(open, 30)
      } else {
        open()
      }
      window.addEventListener("online", reconnectNow)

      return () => {
        if (reconnectOnVisibility) {
          document.removeEventListener("visibilitychange", onVis)
        }
        window.removeEventListener("online", reconnectNow)
        if (retryTimer !== null) clearTimeout(retryTimer)
      }
    }

    let cleanup: (() => void) | undefined
    if (useSse && ssePath) cleanup = connectSse(ssePath)
    else if (wsPath) cleanup = connectWs(wsPath)

    return () => {
      cancelled = true
      cleanup?.()
      const s = sourceRef.current
      if (s) {
        if (s instanceof WebSocket) {
          // Detach handlers first so close events don't trigger a
          // reconnect schedule against a tearing-down hook.
          s.onopen = null
          s.onclose = null
          s.onerror = null
          s.onmessage = null
          if (
            s.readyState === WebSocket.OPEN ||
            s.readyState === WebSocket.CONNECTING
          ) {
            s.close()
          }
        } else {
          s.close()
        }
        sourceRef.current = null
      }
    }
  }, [ssePath, wsPath, reconnectOnVisibility])

  return { state, status }
}
