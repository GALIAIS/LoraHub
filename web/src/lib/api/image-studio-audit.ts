import { ApiError, http } from "./core"

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
