import { http, ApiError } from "./core"

// --------------------------------------------------------------------------- //
// Image Studio
// --------------------------------------------------------------------------- //

export interface ImageStudioItem {
  path: string
  relativePath: string
  name: string
  width: number | null
  height: number | null
  bytes: number
  mtime: number
  caption: string | null
  captionExists: boolean
  annotation: ImageStudioAnnotation | null
  thumbUrl: string
}

export interface ImageStudioAnnotation {
  aiCaption: string | null
  aiQualityScore: number | null
  aiQualityLabel: string | null
  aiQualityReason: string | null
  aiComposition: string | null
  aiTriggerWords: string[] | null
  userQualityLabel: string | null
  userNotes: string | null
  softDeleted: boolean
  favorite: boolean
}

export interface ImageStudioListResponse {
  path: string
  total: number
  page: number
  limit: number
  items: ImageStudioItem[]
}

export interface ImageStudioDetailItem extends ImageStudioItem {
  phash: Record<string, string>
  pendingOps: Array<{
    id: string
    op: string
    payload: Record<string, unknown>
    createdAt: string
  }>
}

export async function imageStudioList(params: {
  path: string
  recursive?: boolean
  page?: number
  limit?: number
  sort?: string
}): Promise<ImageStudioListResponse> {
  const qs = new URLSearchParams({ path: params.path })
  if (params.recursive) qs.set("recursive", "true")
  if (params.page) qs.set("page", String(params.page))
  if (params.limit) qs.set("limit", String(params.limit))
  if (params.sort) qs.set("sort", params.sort)
  return http<ImageStudioListResponse>(`/image-studio/list?${qs}`)
}

export async function imageStudioGetImage(
  path: string,
): Promise<ImageStudioDetailItem> {
  return http<ImageStudioDetailItem>(
    `/image-studio/image?path=${encodeURIComponent(path)}`,
  )
}

export async function imageStudioSaveAnnotation(body: {
  path: string
  userQualityLabel?: string | null
  userNotes?: string | null
  favorite?: boolean
  softDeleted?: boolean
}): Promise<{ ok: boolean; annotation: ImageStudioAnnotation }> {
  return http<{ ok: boolean; annotation: ImageStudioAnnotation }>(
    "/image-studio/annotations",
    {
      method: "PUT",
      body: JSON.stringify(body),
    },
  )
}

export async function imageStudioDeleteAnnotation(
  path: string,
): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(
    `/image-studio/annotations?path=${encodeURIComponent(path)}`,
    { method: "DELETE" },
  )
}

export async function imageStudioAddOp(body: {
  path: string
  op: string
  payload?: Record<string, unknown>
}): Promise<{ id: string; op: string; payload: Record<string, unknown>; createdAt: string }> {
  return http<{
    id: string
    op: string
    payload: Record<string, unknown>
    createdAt: string
  }>("/image-studio/ops", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioListOps(
  path?: string,
): Promise<{ ops: Array<{ id: string; imagePath: string; op: string; payload: Record<string, unknown>; createdAt: string }> }> {
  const qs = path ? `?path=${encodeURIComponent(path)}` : ""
  return http<{
    ops: Array<{
      id: string
      imagePath: string
      op: string
      payload: Record<string, unknown>
      createdAt: string
    }>
  }>(`/image-studio/ops${qs}`)
}

export async function imageStudioDeleteOp(
  opId: string,
): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(`/image-studio/ops/${opId}`, {
    method: "DELETE",
  })
}

export async function imageStudioApplyOps(
  path: string,
): Promise<{ applied: string[]; errors: Array<{ id: string; error: string }> }> {
  return http<{ applied: string[]; errors: Array<{ id: string; error: string }> }>(
    "/image-studio/ops/apply",
    {
      method: "POST",
      body: JSON.stringify({ path }),
    },
  )
}

// --------------------------------------------------------------------------- //
// Image Studio — Dedupe
// --------------------------------------------------------------------------- //

export interface DedupeClusterMember {
  path: string
  hash: string
}

export interface DedupeCluster {
  id: string
  kind: string
  members: DedupeClusterMember[]
  suggestedKeep: string
}

export async function imageStudioDedupeScan(body: {
  path: string
  recursive?: boolean
  algo?: string
  threshold?: number
}): Promise<{ computed: number; total: number; errors: Array<{ path: string; error: string }> }> {
  return http<{
    computed: number
    total: number
    errors: Array<{ path: string; error: string }>
  }>("/image-studio/dedupe/scan", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioDedupeClusters(params: {
  path: string
  kind?: string
  threshold?: number
}): Promise<{ clusters: DedupeCluster[] }> {
  const qs = new URLSearchParams({ path: params.path })
  if (params.kind) qs.set("kind", params.kind)
  if (params.threshold != null) qs.set("threshold", String(params.threshold))
  return http<{ clusters: DedupeCluster[] }>(
    `/image-studio/dedupe/clusters?${qs}`,
  )
}

export async function imageStudioBatchDelete(body: {
  paths: string[]
  forceFavorites?: boolean
}): Promise<{ deletedCount: number; deleted: string[]; bytesFreed: number; errors: Array<{ path: string; error: string }> }> {
  return http<{
    deletedCount: number
    deleted: string[]
    bytesFreed: number
    errors: Array<{ path: string; error: string }>
  }>("/image-studio/dedupe/batch-delete", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

// --------------------------------------------------------------------------- //
// Image Studio — Smart Caption / Tagging from Studio
// --------------------------------------------------------------------------- //

export async function imageStudioSmartCaption(params: {
  path: string
  recursive?: boolean
  device?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  /**
   * "vlm"  — multimodal LLM sees the image directly (default).
   * "tags" — LLM only sees the WD14 tag list, never the image. Use
   *          when the configured VLM is rate-limited / quota-exhausted
   *          or when a cheaper text-only LLM is preferred.
   */
  captionSource?: "vlm" | "tags"
  triggerWord?: string
  stripStyleTags?: boolean
  /** Skip images that already have a non-empty .txt sidecar. Default true. */
  skipExisting?: boolean
  /** Optional progress callback. Fires after each poll with the latest snapshot. */
  onProgress?: (snap: {
    processed: number
    total: number
    percent: number
    last_image: string
    status: string
  }) => void
}): Promise<{ processed: number; results: unknown[]; errors: unknown[] }> {
  // Background-task adapter (uvicorn hang fix): the server now returns
  // 202 with a session_id; we poll status/<id> until it reaches a
  // terminal state, then resolve with the legacy shape so existing
  // callers keep working without code changes.
  const { onProgress, ...body } = params
  const submit = await http<{
    session_id: string
    total: number
    status_url: string
  }>("/image-studio/ai/smart-caption", {
    method: "POST",
    body: JSON.stringify(body),
  })
  const id = submit.session_id
  const poll = async (): Promise<{
    processed: number
    total: number
    percent: number
    last_image: string
    status: string
    results: unknown[]
    errors: unknown[]
    error: string | null
  }> => {
    return http(`/image-studio/ai/smart-caption/status/${encodeURIComponent(id)}`)
  }
  // Simple linear back-off — polls every 1s for first minute, then 3s.
  // Total runtime is bounded by the user explicitly cancelling or by
  // the page being closed (background thread finishes regardless).
  let elapsed = 0
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const snap = await poll()
    if (onProgress) {
      onProgress({
        processed: snap.processed,
        total: snap.total,
        percent: snap.percent,
        last_image: snap.last_image,
        status: snap.status,
      })
    }
    if (snap.status !== "running") {
      if (snap.status === "failed") {
        throw new Error(snap.error ?? "smart-caption batch failed")
      }
      return {
        processed: snap.processed,
        results: snap.results,
        errors: snap.errors,
      }
    }
    const interval = elapsed < 60 ? 1000 : 3000
    await new Promise<void>((resolve) => setTimeout(resolve, interval))
    elapsed += interval / 1000
  }
}

/**
 * Fire-and-return-handle variant of {@link imageStudioSmartCaption}.
 *
 * Returns immediately once the server has accepted the request and
 * issued a session_id. Polling is the caller's responsibility — the
 * studio task store uses this so it can drive a single shared poll
 * loop instead of one per in-flight component.
 */
export async function startSmartCaptionSession(params: {
  path: string
  recursive?: boolean
  device?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  captionSource?: "vlm" | "tags"
  triggerWord?: string
  stripStyleTags?: boolean
  skipExisting?: boolean
}): Promise<{ session_id: string; total: number; status_url: string }> {
  return http<{ session_id: string; total: number; status_url: string }>(
    "/image-studio/ai/smart-caption",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export async function imageStudioSmartCaptionSingle(params: {
  path: string
  device?: string
  captionMode?: "general" | "style" | "character"
  captionSource?: "vlm" | "tags"
  triggerWord?: string
  stripStyleTags?: boolean
  mergeStrategy?: string
}): Promise<{ caption: string; tags: string }> {
  return http<{ caption: string; tags: string }>(
    "/image-studio/ai/smart-caption/single",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export async function imageStudioBatchCaption(params: {
  path: string
  recursive?: boolean
  task?: string
  mergeStrategy?: string
  /** Skip images that already have a non-empty .txt sidecar. Default true. */
  skipAnnotated?: boolean
}): Promise<{ processed: number; skipped?: number; results: unknown[]; errors: unknown[] }> {
  return http<{ processed: number; results: unknown[]; errors: unknown[] }>(
    "/image-studio/ai/caption",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export async function imageStudioBatchQuality(params: {
  path: string
  recursive?: boolean
  task?: string
  /** Skip images that already have an AI quality score. Default true. */
  skipScored?: boolean
}): Promise<{ processed: number; skipped?: number; results: unknown[]; errors: unknown[] }> {
  return http<{ processed: number; skipped?: number; results: unknown[]; errors: unknown[] }>(
    "/image-studio/ai/quality",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export interface ImageStudioTriggerWordsResult {
  processed: number
  skipped?: number
  results: { path: string; triggers: string[] }[]
  errors: { path: string; error: string }[]
  /** Top-8 trigger phrases across the entire dataset, ranked by occurrence. */
  dataset_top: { trigger: string; count: number }[]
}

export async function imageStudioBatchTriggerWords(params: {
  path: string
  recursive?: boolean
  task?: string
  /** Skip images that already have a stored trigger word suggestion. Default true. */
  skipAnalyzed?: boolean
}): Promise<ImageStudioTriggerWordsResult> {
  return http<ImageStudioTriggerWordsResult>(
    "/image-studio/ai/trigger-words",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

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
