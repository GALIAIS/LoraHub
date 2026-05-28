import { http, API_BASE } from "./core"

export interface DatasetScanResponse {
  path: string
  exists: boolean
  recursive: boolean
  image_files: number
  caption_files: number
  missing_caption_files: string[]
  missing_caption_files_truncated: boolean
  samples: Array<{
    name: string
    path: string
    relative_path: string
    caption_exists: boolean
    caption: string | null
  }>
  limit: number
  offset: number
}

export interface DatasetCaptionResponse {
  path: string
  caption: string | null
}

// --------------------------------------------------------------------------- //
// Dataset management (image-studio)
// --------------------------------------------------------------------------- //

export interface DatasetInfo {
  name: string
  path: string
  imageCount: number
  coverPath: string | null
  coverUrl: string | null
  meta: {
    name?: string
    description?: string
    targetResolution?: string
    triggerWord?: string
  }
}

export interface DatasetListResponse {
  root: string
  datasets: DatasetInfo[]
}

export async function datasetList(): Promise<DatasetListResponse> {
  return http<DatasetListResponse>("/image-studio/datasets")
}

export async function datasetCreate(body: {
  name: string
  description?: string
  targetResolution?: string
  triggerWord?: string
}): Promise<{ ok: boolean; path: string; meta: Record<string, string> }> {
  return http<{ ok: boolean; path: string; meta: Record<string, string> }>(
    "/image-studio/datasets",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  )
}

export async function datasetGetMeta(name: string): Promise<Record<string, string>> {
  return http<Record<string, string>>(
    `/image-studio/datasets/${encodeURIComponent(name)}/meta`,
  )
}

export async function datasetUpdateMeta(
  name: string,
  body: { description?: string; targetResolution?: string; triggerWord?: string },
): Promise<{ ok: boolean; meta: Record<string, string> }> {
  return http<{ ok: boolean; meta: Record<string, string> }>(
    `/image-studio/datasets/${encodeURIComponent(name)}/meta`,
    {
      method: "PUT",
      body: JSON.stringify(body),
    },
  )
}

export async function datasetDelete(name: string): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(
    `/image-studio/datasets/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  )
}

export interface UploadProgressEvent {
  file: string
  index: number
  total: number
  status: string
}

export interface UploadCompleteEvent {
  totalExtracted: number
  errors: string[]
}

export function datasetUpload(
  name: string,
  files: File[],
  opts: { keepCaptions?: boolean; onConflict?: string } = {},
): {
  eventSource: ReadableStream<{ event: string; data: unknown }>
  abort: () => void
} {
  const formData = new FormData()
  for (const f of files) {
    formData.append("files", f)
  }
  formData.append("keepCaptions", String(opts.keepCaptions ?? true))
  formData.append("onConflict", opts.onConflict ?? "rename")

  const controller = new AbortController()

  const stream = new ReadableStream<{ event: string; data: unknown }>({
    async start(ctrl) {
      try {
        const r = await fetch(
          `${API_BASE}/image-studio/datasets/${encodeURIComponent(name)}/upload`,
          { method: "POST", body: formData, signal: controller.signal },
        )
        if (!r.ok || !r.body) {
          ctrl.enqueue({ event: "error", data: { message: `upload failed: ${r.status}` } })
          ctrl.close()
          return
        }
        const reader = r.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() || ""
          let currentEvent = ""
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim()
            } else if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6))
                ctrl.enqueue({ event: currentEvent, data })
              } catch { /* skip malformed */ }
            }
          }
        }
        ctrl.close()
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          ctrl.enqueue({ event: "error", data: { message: String(e) } })
        }
        ctrl.close()
      }
    },
  })

  return { eventSource: stream, abort: () => controller.abort() }
}
