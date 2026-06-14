import { http } from "./core"

export interface ShipLintIssue {
  severity: "block" | "warn"
  code: string
  message: string
  count: number
}

export interface ShipLintReport {
  ready: boolean
  stale: boolean
  stale_reason: string | null
  scanned_at: string | null
  image_count: number
  captioned_count: number
  trigger_word: string | null
  trigger_word_hits: number
  issues: ShipLintIssue[]
  blockers: number
  warnings: number
}

export async function imageStudioShipLint(
  datasetPath: string,
): Promise<ShipLintReport> {
  return http(
    `/image-studio/ship/lint?dataset_path=${encodeURIComponent(datasetPath)}`,
  )
}

export async function imageStudioShipExport(body: {
  dataset_path: string
  include_backups?: boolean
  include_quarantine?: boolean
  include_meta?: boolean
  paths?: string[] | null
}): Promise<Response> {
  const resp = await fetch("/api/image-studio/ship/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  return resp
}

export async function imageStudioShipSaveAs(body: {
  source_path: string
  new_name: string
  include_backups?: boolean
  include_quarantine?: boolean
  paths?: string[] | null
}): Promise<{
  ok: boolean
  path: string
  files_copied: number
  images_copied: number
  meta: Record<string, unknown>
}> {
  return http("/image-studio/ship/save-as", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
