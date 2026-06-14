import { http, ApiError } from "./core"
export * from "./image-studio-ai"
export * from "./image-studio-core"
export * from "./image-studio-dedupe"

// --------------------------------------------------------------------------- //
// Image Studio
// --------------------------------------------------------------------------- //

// ─── Image Studio: Audit ─────────────────────────────────────────────

export interface AuditHistogramBucket {
  bucket: string
  count: number
}

export interface AuditTagRow {
  tag: string
  count: number
}

export type AuditIssueKind =
  | "corrupt"
  | "tiny"
  | "exif_rotation"
  | "no_caption"
  | "missing_trigger"
  | "blurry"

export interface AuditIssue {
  kind: AuditIssueKind
  path: string
  // Each kind carries its own extras (msg / width / score / etc.)
  // Lift them with bracket access in the UI.
  [key: string]: unknown
}

export interface AuditReport {
  dataset_path: string
  scanned_at: string
  image_count: number
  captioned_count: number
  trigger_word: string | null
  trigger_word_hits: number
  resolution_histogram: AuditHistogramBucket[]
  ar_histogram: AuditHistogramBucket[]
  filesize_histogram: AuditHistogramBucket[]
  caption_length_histogram: AuditHistogramBucket[]
  tag_vocab: AuditTagRow[]
  issues: AuditIssue[]
  duration_s: number
}

export async function imageStudioAuditScan(body: {
  dataset_path: string
  recursive?: boolean
  trigger_word?: string | null
  blur_check?: boolean
  max_images?: number | null
}): Promise<AuditReport> {
  return http<AuditReport>("/image-studio/audit/scan", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioAuditReport(
  datasetPath: string,
): Promise<AuditReport | null> {
  // 404 → no cache yet (caller renders empty state).
  try {
    return await http<AuditReport>(
      `/image-studio/audit/report?dataset_path=${encodeURIComponent(
        datasetPath,
      )}`,
    )
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null
    throw e
  }
}

// ─── Image Studio: Curate ────────────────────────────────────────────

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

// ─── Image Studio: Captions (vocab + batch edit) ─────────────────────

export interface CaptionVocabRow {
  tag: string
  count: number
}

export async function imageStudioCaptionsVocab(
  datasetPath: string,
  opts: { recursive?: boolean; limit?: number; case_sensitive?: boolean } = {},
): Promise<{
  files_seen: number
  tag_count: number
  vocab: CaptionVocabRow[]
}> {
  const qs = new URLSearchParams({ dataset_path: datasetPath })
  if (opts.recursive !== undefined) qs.set("recursive", String(opts.recursive))
  if (opts.limit !== undefined) qs.set("limit", String(opts.limit))
  if (opts.case_sensitive !== undefined)
    qs.set("case_sensitive", String(opts.case_sensitive))
  return http(`/image-studio/captions/vocab?${qs}`)
}

export interface CaptionDiff {
  path: string
  caption_path: string
  before: string
  after: string
  matches: number
}

export async function imageStudioCaptionsFindReplace(body: {
  dataset_path: string
  pattern: string
  replacement?: string
  is_regex?: boolean
  case_sensitive?: boolean
  whole_caption?: boolean
  dry_run?: boolean
  recursive?: boolean
  paths?: string[] | null
}): Promise<{
  dry_run: boolean
  matched_files: number
  matched_count: number
  diffs: CaptionDiff[]
  diffs_truncated: boolean
  written: string[]
}> {
  return http("/image-studio/captions/find-replace", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioCaptionsInjectTrigger(body: {
  dataset_path: string
  trigger_word: string
  position?: "prepend" | "append"
  skip_existing?: boolean
  recursive?: boolean
  paths?: string[] | null
}): Promise<{
  trigger: string
  position: string
  injected_count: number
  skipped_count: number
}> {
  return http("/image-studio/captions/inject-trigger", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioCaptionsBlacklist(body: {
  dataset_path: string
  tags: string[]
  case_sensitive?: boolean
  recursive?: boolean
  paths?: string[] | null
}): Promise<{
  edited_count: number
  removed_count: number
  blacklisted_tags: string[]
}> {
  return http("/image-studio/captions/blacklist", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

// ─── Image Studio: Ship (training-readiness + export + save-as) ───────

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

/** Trigger a dataset zip download via the streaming export endpoint.
 *  Returns the underlying Response so the caller can pipe it into a
 *  `<a download>` ObjectURL or pass to a service worker. */
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

// ─── Image Studio: Intake (server-side import) ───────────────────────

export interface IntakePreflightFile {
  source_path: string
  phash: string | null
}

export interface IntakePreflightReport {
  candidate_count: number
  new_count: number
  duplicate_existing_count: number
  duplicate_within_batch_count: number
  new: IntakePreflightFile[]
  duplicate_existing: IntakePreflightFile[]
  duplicate_within_batch: IntakePreflightFile[]
  truncated: boolean
}

export async function imageStudioIntakePreflight(body: {
  dataset_path: string
  source_path: string
  recursive?: boolean
  phash_threshold?: number
}): Promise<IntakePreflightReport> {
  return http("/image-studio/intake/preflight", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export interface IntakeResult {
  imported_count: number
  skipped_count: number
  failed_count: number
  imported: { source_path: string; imported_path: string }[]
  skipped: { source_path: string; reason: string }[]
  failed: { source_path: string; error: string }[]
}

export async function imageStudioIntakeLocalPath(body: {
  dataset_path: string
  source_path: string
  recursive?: boolean
  skip_duplicates?: boolean
  phash_threshold?: number
  move?: boolean
}): Promise<IntakeResult> {
  return http("/image-studio/intake/local-path", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioIntakeFromDataset(body: {
  dataset_path: string
  source_dataset_path: string
  pattern?: string
  skip_duplicates?: boolean
  phash_threshold?: number
}): Promise<IntakeResult & { candidate_count: number }> {
  return http("/image-studio/intake/from-dataset", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
