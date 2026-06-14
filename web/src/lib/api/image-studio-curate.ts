import { http } from "./core"
import type { AuditIssueKind } from "./image-studio-audit"

export interface QuarantineEntry {
  moved_at: string
  original_path: string
  quarantine_path: string
  caption_quarantine_path: string | null
  reason: string | null
  restored_at?: string
  restored_path?: string
}

export async function imageStudioAutoRotate(body: {
  dataset_path: string
  paths?: string[]
  recursive?: boolean
}): Promise<{
  rotated: string[]
  rotated_count: number
  skipped_count: number
  failed: { path: string; error: string }[]
}> {
  return http("/image-studio/curate/auto-rotate", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export interface ImageStudioAutoRotateSession {
  session_id: string
  dataset_path: string
  status: string
  processed: number
  total: number
  percent: number
  last_image: string
  rotated: string[]
  rotated_count: number
  skipped_count: number
  failed: { path: string; error: string }[]
  error: string | null
  started_at: number
  finished_at: number | null
}

export async function startImageStudioAutoRotate(body: {
  dataset_path: string
  paths?: string[]
  recursive?: boolean
}): Promise<{ session_id: string; total: number; status_url: string }> {
  return http<{ session_id: string; total: number; status_url: string }>(
    "/image-studio/curate/auto-rotate/start",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  )
}

export async function getImageStudioAutoRotateSession(
  id: string,
): Promise<ImageStudioAutoRotateSession> {
  return http<ImageStudioAutoRotateSession>(
    `/image-studio/curate/auto-rotate/status/${encodeURIComponent(id)}`,
  )
}

export async function imageStudioQuarantine(body: {
  dataset_path: string
  paths: string[]
  reason?: string | null
}): Promise<{
  moved: QuarantineEntry[]
  moved_count: number
  failed: { path: string; error: string }[]
}> {
  return http("/image-studio/curate/quarantine", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioQuarantineList(
  datasetPath: string,
): Promise<{ entries: QuarantineEntry[] }> {
  return http(
    `/image-studio/curate/quarantine?dataset_path=${encodeURIComponent(datasetPath)}`,
  )
}

export async function imageStudioRestoreQuarantine(body: {
  dataset_path: string
  quarantine_paths: string[]
}): Promise<{
  restored: QuarantineEntry[]
  restored_count: number
  failed: { path: string; error: string }[]
}> {
  return http("/image-studio/curate/restore-quarantine", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioBatchResize(body: {
  dataset_path: string
  paths?: string[]
  target_short_edge: number
  filter?: "lanczos" | "bicubic" | "bilinear"
  upscale?: boolean
  recursive?: boolean
}): Promise<{
  resampled: { path: string; from: [number, number]; to: [number, number] }[]
  resampled_count: number
  skipped_count: number
  failed: { path: string; error: string }[]
}> {
  return http("/image-studio/curate/batch-resize", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export interface ImageStudioBatchResizeSession {
  session_id: string
  dataset_path: string
  status: string
  processed: number
  total: number
  percent: number
  last_image: string
  resampled: { path: string; from: [number, number]; to: [number, number] }[]
  resampled_count: number
  skipped_count: number
  failed: { path: string; error: string }[]
  error: string | null
  started_at: number
  finished_at: number | null
}

export async function startImageStudioBatchResize(body: {
  dataset_path: string
  paths?: string[]
  target_short_edge: number
  filter?: "lanczos" | "bicubic" | "bilinear"
  upscale?: boolean
  recursive?: boolean
}): Promise<{ session_id: string; total: number; status_url: string }> {
  return http<{ session_id: string; total: number; status_url: string }>(
    "/image-studio/curate/batch-resize/start",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  )
}

export async function getImageStudioBatchResizeSession(
  id: string,
): Promise<ImageStudioBatchResizeSession> {
  return http<ImageStudioBatchResizeSession>(
    `/image-studio/curate/batch-resize/status/${encodeURIComponent(id)}`,
  )
}

export async function imageStudioBatchByIssue(body: {
  dataset_path: string
  issue_kinds: AuditIssueKind[]
  action: "quarantine" | "delete"
  reason?: string | null
}): Promise<{
  action: string
  matched_count: number
  result: {
    moved: QuarantineEntry[]
    moved_count: number
    failed: { path: string; error: string }[]
  } | null
}> {
  return http("/image-studio/curate/batch-by-issue", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export interface BackupEntry {
  backup_path: string
  relative_path: string
  size: number
  mtime: number
}

export async function imageStudioBackupsList(
  datasetPath: string,
): Promise<{ entries: BackupEntry[] }> {
  return http(
    `/image-studio/curate/backups?dataset_path=${encodeURIComponent(datasetPath)}`,
  )
}

export async function imageStudioRestoreBackup(body: {
  dataset_path: string
  backup_paths: string[]
}): Promise<{
  restored: string[]
  restored_count: number
  failed: { path: string; error: string }[]
}> {
  return http("/image-studio/curate/restore-backup", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
