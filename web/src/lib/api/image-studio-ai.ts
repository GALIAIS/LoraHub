import { http } from "./core"

export async function imageStudioSmartCaption(params: {
  path: string
  recursive?: boolean
  device?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  promptTemplate?: string
  /**
   * "vlm"  — multimodal LLM sees the image directly (default).
   * "tags" — LLM only sees the WD14 tag list, never the image. Use
   *          when the configured VLM is rate-limited / quota-exhausted
   *          or when a cheaper text-only LLM is preferred.
   */
  captionSource?: "vlm" | "tags" | "toriigate"
  triggerWord?: string
  stripStyleTags?: boolean
  useWd14?: boolean
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

export async function startSmartCaptionSession(params: {
  path: string
  recursive?: boolean
  device?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  promptTemplate?: string
  captionSource?: "vlm" | "tags" | "toriigate"
  triggerWord?: string
  stripStyleTags?: boolean
  useWd14?: boolean
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
  promptTemplate?: string
  captionSource?: "vlm" | "tags" | "toriigate"
  triggerWord?: string
  stripStyleTags?: boolean
  useWd14?: boolean
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

export interface Wd14PrefilterResult {
  path: string
  ratingName: string | null
  generalTags: string[]
  characterTags: string[]
  promptText: string
  dataUrl: string
  captionSource: "vlm" | "tags" | "toriigate"
  stripStyleTags: boolean
  skipLlm: boolean
}

export async function imageStudioWd14Prefilter(params: {
  path: string
  taggerModel?: string
  device?: string
  generalThreshold?: number
  characterThreshold?: number
  captionMode?: "general" | "style" | "character"
  promptTemplate?: string
  captionSource?: "vlm" | "tags" | "toriigate"
  triggerWord?: string
  stripStyleTags?: boolean
}): Promise<Wd14PrefilterResult> {
  return http<Wd14PrefilterResult>("/image-studio/ai/wd14-prefilter", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function imageStudioVlmAnimaRewrite(params: {
  path: string
  visionTask?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  promptTemplate?: string
  captionSource?: "vlm" | "tags" | "toriigate"
  triggerWord?: string
  stripStyleTags?: boolean
  ratingName?: string | null
  generalTags?: string[]
  characterTags?: string[]
  promptText: string
  dataUrl?: string
  skipLlm?: boolean
}): Promise<{ ok: true; path: string; wd14Tags: string; caption: string }> {
  return http<{ ok: true; path: string; wd14Tags: string; caption: string }>(
    "/image-studio/ai/vlm-anima-rewrite",
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
}): Promise<{
  processed: number
  skipped?: number
  results: unknown[]
  errors: unknown[]
}> {
  return http<{ processed: number; results: unknown[]; errors: unknown[] }>(
    "/image-studio/ai/caption",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export interface ImageStudioCaptionSession {
  session_id: string
  path: string
  status: string
  processed: number
  total: number
  skipped: number
  percent: number
  last_image: string
  results: { path: string; caption: string }[]
  errors: { path: string; error: string }[]
  error: string | null
  started_at: number
  finished_at: number | null
}

export async function startCaptionSession(params: {
  path: string
  recursive?: boolean
  task?: string
  mergeStrategy?: string
  /** Skip images that already have a non-empty .txt sidecar. Default true. */
  skipAnnotated?: boolean
}): Promise<{
  session_id: string
  total: number
  skipped: number
  status_url: string
}> {
  return http<{
    session_id: string
    total: number
    skipped: number
    status_url: string
  }>("/image-studio/ai/caption/start", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function getCaptionSession(
  id: string,
): Promise<ImageStudioCaptionSession> {
  return http<ImageStudioCaptionSession>(
    `/image-studio/ai/caption/status/${encodeURIComponent(id)}`,
  )
}

export async function imageStudioBatchQuality(params: {
  path: string
  recursive?: boolean
  task?: string
  /** Skip images that already have an AI quality score. Default true. */
  skipScored?: boolean
}): Promise<{
  processed: number
  skipped?: number
  results: unknown[]
  errors: unknown[]
}> {
  return http<{
    processed: number
    skipped?: number
    results: unknown[]
    errors: unknown[]
  }>("/image-studio/ai/quality", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export interface ImageStudioQualitySession {
  session_id: string
  path: string
  status: string
  processed: number
  total: number
  skipped: number
  percent: number
  last_image: string
  results: unknown[]
  errors: { path: string; error: string }[]
  error: string | null
  started_at: number
  finished_at: number | null
}

export async function startQualitySession(params: {
  path: string
  recursive?: boolean
  task?: string
  /** Skip images that already have an AI quality score. Default true. */
  skipScored?: boolean
}): Promise<{
  session_id: string
  total: number
  skipped: number
  status_url: string
}> {
  return http<{
    session_id: string
    total: number
    skipped: number
    status_url: string
  }>("/image-studio/ai/quality/start", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function getQualitySession(
  id: string,
): Promise<ImageStudioQualitySession> {
  return http<ImageStudioQualitySession>(
    `/image-studio/ai/quality/status/${encodeURIComponent(id)}`,
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

export interface ImageStudioTriggerWordsSession {
  session_id: string
  path: string
  status: string
  processed: number
  total: number
  skipped: number
  percent: number
  last_image: string
  results: { path: string; triggers: string[] }[]
  errors: { path: string; error: string }[]
  dataset_top: { trigger: string; count: number }[]
  error: string | null
  started_at: number
  finished_at: number | null
}

export async function startTriggerWordsSession(params: {
  path: string
  recursive?: boolean
  task?: string
  /** Skip images that already have a stored trigger word suggestion. Default true. */
  skipAnalyzed?: boolean
}): Promise<{
  session_id: string
  total: number
  skipped: number
  status_url: string
}> {
  return http<{
    session_id: string
    total: number
    skipped: number
    status_url: string
  }>("/image-studio/ai/trigger-words/start", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function getTriggerWordsSession(
  id: string,
): Promise<ImageStudioTriggerWordsSession> {
  return http<ImageStudioTriggerWordsSession>(
    `/image-studio/ai/trigger-words/status/${encodeURIComponent(id)}`,
  )
}
