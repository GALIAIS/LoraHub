import { http } from "./core"

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
}): Promise<{
  id: string
  op: string
  payload: Record<string, unknown>
  createdAt: string
}> {
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

export async function imageStudioListOps(path?: string): Promise<{
  ops: Array<{
    id: string
    imagePath: string
    op: string
    payload: Record<string, unknown>
    createdAt: string
  }>
}> {
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
