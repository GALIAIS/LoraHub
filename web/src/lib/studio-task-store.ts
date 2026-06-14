/**
 * Global tracker for image-studio background AI tasks.
 *
 * Why a hand-rolled store and not TanStack Query / useState:
 * - The image-studio dataset-detail page is lazy-loaded; navigating away
 *   unmounts it, which throws away every local useState (including the
 *   AI progress banner) and tears down any setInterval polling.
 * - The actual work runs server-side via background tasks: callers POST
 *   to start, get back a session_id, and poll a status endpoint until
 *   it reaches a terminal state. Killing the polling on unmount makes
 *   the user think the task died, even though it's still running.
 * - TanStack Query keys are scoped to mounted components by default,
 *   so the cache stays around but no observer ticks the refetch loop.
 *
 * This module owns a single in-process store that:
 *   1. Tracks active sessions (id, kind, dataset, progress, status)
 *   2. Drives a single setInterval polling loop independent of React
 *   3. Persists session ids to localStorage so a page reload can pick
 *      back up sessions that are still alive on the server
 *   4. Exposes useSyncExternalStore-based hooks any component can
 *      subscribe to without coupling to mount lifecycle
 *
 * Two endpoint flavours are supported:
 *   - "sessioned" (caption, smart-caption, quality-score, trigger-words, wd14): server returns session_id,
 *     status endpoint exists, full reconnect after reload works
 *   - "in-flight": synchronous endpoints
 *     with no session_id, so the only thing we can do is keep the
 *     promise alive across navigations within the same page-load.
 *     These records are NOT persisted to localStorage (a page reload
 *     would orphan them anyway — we have no way to find them again).
 */
import {
  getCaptionSession,
  getQualitySession,
  getTaggingSession,
  getTriggerWordsSession,
  http,
  type TaggingSession,
} from "@/lib/api"

export type StudioTaskKind =
  | "caption"
  | "smart-caption"
  | "wd14"
  | "trigger-words"
  | "quality-score"

export type StudioTaskStatus =
  | "running"
  | "completed"
  | "failed"
  | "cancelled"

export interface StudioTaskRecord {
  /** Stable id. Server-issued session_id when available, otherwise a synthetic uuid for in-flight kinds. */
  id: string
  kind: StudioTaskKind
  /** Absolute dataset path the task is operating on — used to scope per-page progress banners. */
  datasetPath: string
  /** User-facing label rendered in progress banners. */
  label: string
  /** Epoch ms of when addTask was called. */
  startedAt: number
  status: StudioTaskStatus
  processed?: number
  total?: number
  /** Most recent image filename touched, for the "智能标注中… · foo.png" hint. */
  lastImage?: string
  /** Final error message when status === "failed". */
  errorMsg?: string
  /** True for kinds where the server gives us a session_id we can poll/persist. */
  sessioned: boolean
  /** Consecutive poll error counter — three strikes and we mark the task failed. */
  pollErrors?: number
}

const STORAGE_KEY = "lorahub.studio.tasks"
/** Drop persisted records older than this when we boot — server sessions don't outlive a day in practice. */
const TTL_MS = 24 * 60 * 60 * 1000
/** Single shared poll cadence. Two seconds matches the WD14 poll the inline interval used. */
const POLL_INTERVAL_MS = 2000
/** How many consecutive failed polls before we declare the task dead. */
const POLL_ERROR_LIMIT = 3

type Listener = () => void

const listeners = new Set<Listener>()
let tasks: StudioTaskRecord[] = []
let pollTimer: ReturnType<typeof setInterval> | null = null

// --------------------------------------------------------------------------- //
// Persistence
// --------------------------------------------------------------------------- //

interface PersistedRecord {
  id: string
  kind: StudioTaskKind
  datasetPath: string
  label: string
  startedAt: number
  status: StudioTaskStatus
  processed?: number
  total?: number
  lastImage?: string
  errorMsg?: string
}

function load(): StudioTaskRecord[] {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: PersistedRecord[] = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    const now = Date.now()
    return parsed
      .filter((r) => r && typeof r.id === "string")
      .filter((r) => now - (r.startedAt ?? 0) <= TTL_MS)
      .map((r) => ({ ...r, sessioned: true, pollErrors: 0 }))
  } catch {
    return []
  }
}

function persist(): void {
  if (typeof window === "undefined") return
  try {
    // Only sessioned records can be revived after a reload. In-flight
    // (synchronous-endpoint) tasks have no server-side handle to reconnect
    // to, so persisting them would just bloat localStorage with corpses.
    const persistable = tasks
      .filter((t) => t.sessioned)
      .map(({ pollErrors: _ignored, sessioned: _ignored2, ...rest }) => rest)
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable))
  } catch {
    // localStorage can be full / disabled in private mode — non-fatal.
  }
}

function emit(): void {
  for (const l of listeners) l()
}

function commit(next: StudioTaskRecord[]): void {
  tasks = next
  persist()
  emit()
  ensurePolling()
}

// --------------------------------------------------------------------------- //
// Polling
// --------------------------------------------------------------------------- //

function ensurePolling(): void {
  const hasRunning = tasks.some((t) => t.status === "running" && t.sessioned)
  if (hasRunning && pollTimer == null) {
    pollTimer = setInterval(() => void tick(), POLL_INTERVAL_MS)
  } else if (!hasRunning && pollTimer != null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function tick(): Promise<void> {
  const targets = tasks.filter((t) => t.status === "running" && t.sessioned)
  if (targets.length === 0) return
  await Promise.all(targets.map((t) => pollOne(t)))
}

async function pollOne(task: StudioTaskRecord): Promise<void> {
  try {
    if (task.kind === "caption") {
      const snap = await getCaptionSession(task.id)
      const status = mapSmartCaptionStatus(snap.status)
      updateTask(task.id, {
        processed: snap.processed,
        total: snap.total,
        lastImage: snap.last_image || undefined,
        status,
        errorMsg: snap.error ?? undefined,
        pollErrors: 0,
      })
    } else if (task.kind === "smart-caption") {
      const snap = await http<{
        processed: number
        total: number
        percent: number
        last_image: string
        status: string
        error: string | null
      }>(
        `/image-studio/ai/smart-caption/status/${encodeURIComponent(task.id)}`,
      )
      const status = mapSmartCaptionStatus(snap.status)
      updateTask(task.id, {
        processed: snap.processed,
        total: snap.total,
        lastImage: snap.last_image || undefined,
        status,
        errorMsg: snap.error ?? undefined,
        pollErrors: 0,
      })
    } else if (task.kind === "quality-score") {
      const snap = await getQualitySession(task.id)
      const status = mapSmartCaptionStatus(snap.status)
      updateTask(task.id, {
        processed: snap.processed,
        total: snap.total,
        lastImage: snap.last_image || undefined,
        status,
        errorMsg: snap.error ?? undefined,
        pollErrors: 0,
      })
    } else if (task.kind === "trigger-words") {
      const snap = await getTriggerWordsSession(task.id)
      const status = mapSmartCaptionStatus(snap.status)
      updateTask(task.id, {
        processed: snap.processed,
        total: snap.total,
        lastImage: snap.last_image || undefined,
        status,
        errorMsg: snap.error ?? undefined,
        pollErrors: 0,
      })
    } else if (task.kind === "wd14") {
      const snap: TaggingSession = await getTaggingSession(task.id)
      const status = mapWd14Status(snap.status)
      updateTask(task.id, {
        processed: snap.written,
        total: snap.total ?? undefined,
        status,
        errorMsg: snap.error ?? undefined,
        pollErrors: 0,
      })
    }
  } catch (err) {
    const next = (task.pollErrors ?? 0) + 1
    if (next >= POLL_ERROR_LIMIT) {
      // Three strikes: assume the session is gone (server restart, expired,
      // or 404). Mark failed so the banner stops claiming "running" forever.
      updateTask(task.id, {
        status: "failed",
        errorMsg:
          err instanceof Error ? err.message : "状态轮询连续失败 3 次",
        pollErrors: next,
      })
    } else {
      updateTask(task.id, { pollErrors: next })
    }
  }
}

function mapSmartCaptionStatus(s: string): StudioTaskStatus {
  switch (s) {
    case "running":
    case "pending":
      return "running"
    case "succeeded":
      return "completed"
    case "canceled":
    case "cancelled":
    case "interrupted":
      return "cancelled"
    case "failed":
      return "failed"
    default:
      return "running"
  }
}

function mapWd14Status(s: string): StudioTaskStatus {
  switch (s) {
    case "running":
      return "running"
    case "succeeded":
      return "completed"
    case "canceled":
    case "cancelled":
    case "interrupted":
      return "cancelled"
    case "failed":
      return "failed"
    default:
      return "running"
  }
}

// --------------------------------------------------------------------------- //
// Public mutators
// --------------------------------------------------------------------------- //

export interface AddTaskInput {
  id: string
  kind: StudioTaskKind
  datasetPath: string
  label: string
  total?: number
  /** Whether the server returns a session_id we can poll. Defaults to true for sessioned kinds. */
  sessioned?: boolean
}

export function addTask(input: AddTaskInput): StudioTaskRecord {
  const sessioned =
    input.sessioned ??
    (input.kind === "caption" ||
      input.kind === "smart-caption" ||
      input.kind === "quality-score" ||
      input.kind === "trigger-words" ||
      input.kind === "wd14")
  const record: StudioTaskRecord = {
    id: input.id,
    kind: input.kind,
    datasetPath: input.datasetPath,
    label: input.label,
    startedAt: Date.now(),
    status: "running",
    total: input.total,
    sessioned,
    pollErrors: 0,
  }
  // Replace any existing record with the same id (re-running a wd14 batch
  // with the same session_id is unusual but keep the latest snapshot).
  const next = tasks.filter((t) => t.id !== record.id)
  next.push(record)
  commit(next)
  return record
}

export function updateTask(
  id: string,
  patch: Partial<Omit<StudioTaskRecord, "id" | "kind" | "datasetPath">>,
): void {
  let changed = false
  const next = tasks.map((t) => {
    if (t.id !== id) return t
    changed = true
    return { ...t, ...patch }
  })
  if (changed) commit(next)
}

export function removeTask(id: string): void {
  const next = tasks.filter((t) => t.id !== id)
  if (next.length !== tasks.length) commit(next)
}

/** Mark all terminal tasks (completed/failed/cancelled) for a dataset as dismissed. */
export function dismissTerminalFor(datasetPath: string): void {
  const next = tasks.filter(
    (t) => !(t.datasetPath === datasetPath && t.status !== "running"),
  )
  if (next.length !== tasks.length) commit(next)
}

// --------------------------------------------------------------------------- //
// Subscription primitives
// --------------------------------------------------------------------------- //

export function subscribe(listener: Listener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function getSnapshot(): StudioTaskRecord[] {
  return tasks
}

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //

// Hydrate from localStorage on module load. Any record whose status is
// already terminal stays in the store as a "dismissable banner" so the
// user who reloaded mid-task still sees the final result on return.
tasks = load()
ensurePolling()
