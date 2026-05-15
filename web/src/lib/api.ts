import { useEffect, useRef, useState } from "react"

const API_BASE = "/api"

export interface JobSummary {
  id: string
  state: string
  workspace: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  returncode: number | null
  error: string | null
  pid: number | null
}

export interface TrainingEvent {
  type: string
  payload: Record<string, unknown>
  timestamp: number
  job_id: string | null
}

export interface RecipeListEntry {
  name: string
  filename: string
  size: number
  valid: boolean
  arch: string | null
  summary: string | null
  error: string | null
}

export interface RecipeDetail {
  name: string
  filename: string
  path: string
  content: string
  parsed: Record<string, unknown> | null
  error: string | null
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => http<{ status: string; version: string }>("/health"),
  listJobs: () => http<{ jobs: JobSummary[] }>("/jobs"),
  getJob: (id: string) => http<JobSummary>(`/jobs/${id}`),
  getEvents: (id: string, limit = 200) =>
    http<{ events: TrainingEvent[] }>(`/jobs/${id}/events?limit=${limit}`),
  cancelJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}`, { method: "DELETE" }),
  recipeSchema: () => http<Record<string, unknown>>("/recipes/schema"),
  listRecipes: () =>
    http<{ dir: string; recipes: RecipeListEntry[] }>("/recipes"),
  getRecipe: (name: string) =>
    http<RecipeDetail>(`/recipes/${encodeURIComponent(name)}`),
  createJob: (recipe: Record<string, unknown>, workspace?: string) =>
    http<JobSummary>("/jobs", {
      method: "POST",
      body: JSON.stringify({ recipe, workspace }),
    }),
}

/**
 * Live event stream over WebSocket. Returns the latest snapshot of buffered
 * events plus the connection state; reconnects on close while the job is alive.
 */
export function useJobStream(jobId: string | null) {
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!jobId) return
    setEvents([])
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.host || "127.0.0.1:18765"
    const ws = new WebSocket(`${protocol}//${host}/api/jobs/${jobId}/stream`)
    wsRef.current = ws

    ws.onopen = () => setStatus("open")
    ws.onclose = () => setStatus("closed")
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as TrainingEvent
        setEvents((prev) => [...prev, event].slice(-500))
      } catch {
        // ignore malformed frames
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [jobId])

  return { events, status }
}
