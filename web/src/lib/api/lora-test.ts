import { http } from "./core"
import type { JobFile } from "./jobs"
import type { TaskSessionRecord } from "./tasks"

export interface LoraTestJob {
  job_id: string
  output_name: string | null
  workspace: string
  state: string
  backend: string | null
  base_model: Record<string, unknown>
  created_at: string | null
  finished_at: string | null
  checkpoints: JobFile[]
}

export interface LoraTestModelsResponse {
  jobs: LoraTestJob[]
}

export interface LoraTestGenerateInput {
  job_id: string
  checkpoint_path: string
  prompt: string
  negative_prompt?: string
  width?: number
  height?: number
  seed?: number
  batch_count?: number
  steps?: number
  cfg?: number
  sampler?: string
  lora_weight?: number
  output_format?: "png"
}

export interface LoraTestGenerateResponse {
  session_id: string
}

export function listLoraTestModels() {
  return http<LoraTestModelsResponse>("/lora-test/models")
}

export function startLoraTestGeneration(input: LoraTestGenerateInput) {
  return http<LoraTestGenerateResponse>("/lora-test/generate", {
    method: "POST",
    body: JSON.stringify(input),
  })
}

export function getLoraTestSession(sessionId: string) {
  return http<TaskSessionRecord>(
    `/lora-test/sessions/${encodeURIComponent(sessionId)}`,
  )
}

export function cancelLoraTestSession(sessionId: string) {
  return http<{ canceled: boolean }>(
    `/lora-test/sessions/${encodeURIComponent(sessionId)}/cancel`,
    { method: "POST" },
  )
}

export function loraTestResultFileUrl(sessionId: string, path: string) {
  return `/api/lora-test/results/${encodeURIComponent(sessionId)}/file?path=${encodeURIComponent(path)}`
}
