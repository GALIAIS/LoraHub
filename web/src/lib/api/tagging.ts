import { http } from "./core"

export interface TaggingEvent {
  ts: number
  message: string
  percent: number
  image: string | null
}

export interface TaggingSession {
  session_id: string
  path: string
  model_id: string
  device: "auto" | "cpu" | "cuda"
  general: number
  character: number
  overwrite: boolean
  recursive: boolean
  include_character: boolean
  underscores: boolean
  status: "running" | "stop_requested" | "succeeded" | "failed" | "canceled" | "interrupted"
  percent: number
  events: TaggingEvent[]
  written: number
  total: number | null
  active_provider: string
  error: string | null
  started_at: number
  finished_at: number | null
}

export interface TagDatasetRequest {
  path: string
  model_id?: string
  general?: number
  character?: number
  device?: "auto" | "cpu" | "cuda"
  overwrite?: boolean
  recursive?: boolean
  include_character?: boolean
  underscores?: boolean
}

// Streaming-ish helpers used by the studio task store. The `api.tagDataset`
// entry on the legacy client object is a separate POST /tagging/tag call
// that returns a full `TaggingSession`; this start variant returns just
// the id + sets up a polling loop in the caller.
export async function startTaggingSession(params: {
  path: string
  tagger?: string
  model_id?: string
  general?: number
  character?: number
  device?: string
  overwrite?: boolean
  recursive?: boolean
}): Promise<{ session_id: string }> {
  return http<{ session_id: string }>("/tagging/tag", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function getTaggingSession(
  sessionId: string,
): Promise<TaggingSession> {
  return http<TaggingSession>(`/tagging/tag/${sessionId}`)
}

export async function stopTaggingSession(
  sessionId: string,
): Promise<{ session_id: string; status: string }> {
  return http<{ session_id: string; status: string }>(
    `/tagging/tag/${encodeURIComponent(sessionId)}/stop`,
    { method: "POST" },
  )
}
