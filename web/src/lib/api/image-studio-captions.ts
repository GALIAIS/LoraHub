import { http } from "./core"

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
