import { http } from "./core"

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
}): Promise<{
  computed: number
  total: number
  errors: Array<{ path: string; error: string }>
}> {
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
}): Promise<{
  deletedCount: number
  deleted: string[]
  bytesFreed: number
  errors: Array<{ path: string; error: string }>
}> {
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

export async function imageStudioSimilarityScan(body: {
  path: string
  recursive?: boolean
  mode?: "embedding" | "pairwise"
  threshold?: number
  task?: string
}): Promise<{
  computed: number
  total: number
  errors: Array<{ path: string; error: string }>
}> {
  return http("/image-studio/similarity/scan", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioSimilarityClusters(params: {
  path: string
  kind?: "ai" | "phash"
  threshold?: number
}): Promise<{ clusters: DedupeCluster[] }> {
  const qs = new URLSearchParams({ path: params.path })
  if (params.kind) qs.set("kind", params.kind)
  if (params.threshold != null) qs.set("threshold", String(params.threshold))
  return http<{ clusters: DedupeCluster[] }>(
    `/image-studio/similarity/clusters?${qs}`,
  )
}

export async function imageStudioSimilarityBatchDelete(body: {
  paths: string[]
  forceFavorites?: boolean
}): Promise<{
  deletedCount: number
  deleted: string[]
  bytesFreed: number
  errors: Array<{ path: string; error: string }>
}> {
  return http("/image-studio/similarity/batch-delete", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
