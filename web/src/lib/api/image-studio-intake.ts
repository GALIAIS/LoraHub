import { http } from "./core"

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
