import { http } from "./core"

export interface TaskSessionEvent {
  level: "info" | "warn" | "error" | string
  message: string
  percent?: number | null
  payload?: Record<string, unknown>
  ts: number
}

export interface TaskSessionRecord {
  id: string
  kind: string
  title: string
  status:
    | "queued"
    | "running"
    | "stop_requested"
    | "succeeded"
    | "failed"
    | "canceled"
    | "interrupted"
    | string
  percent: number
  metadata: Record<string, unknown>
  result: Record<string, unknown> | null
  error: string | null
  started_at: number
  updated_at: number
  finished_at: number | null
  events: TaskSessionEvent[]
}

export async function getLatestTask(
  kind: string,
): Promise<TaskSessionRecord> {
  return http<TaskSessionRecord>(`/tasks/latest?kind=${encodeURIComponent(kind)}`)
}

export async function getTask(sessionId: string): Promise<TaskSessionRecord> {
  return http<TaskSessionRecord>(`/tasks/${encodeURIComponent(sessionId)}`)
}
