import type { BackendId } from "./backends"

export interface ConfigListEntry {
  name: string
  filename: string
  size: number
  modified_at: number
  valid: boolean
  arch: string | null
  /** Which training backend this config targets. Null when load() failed. */
  backend: BackendId | null
  summary: string | null
  error: string | null
}

export interface ConfigDetail {
  name: string
  filename: string
  path: string
  content: string
  parsed: Record<string, unknown> | null
  error: string | null
}

export interface ValidationFieldError {
  loc: (string | number)[]
  msg: string
  type: string
}

export interface ValidateResponse {
  valid: boolean
  normalized?: Record<string, unknown>
  errors?: ValidationFieldError[]
  preflight?: {
    issues: Array<{ severity: string; field: string; message: string }>
    vram: {
      model_mib: number
      optimizer_mib: number
      activations_mib: number
      overhead_mib: number
      total_mib: number
      total_gib: number
    }
    paths: {
      checkpoint_exists: boolean
      dataset_exists: boolean
      image_files: number
      caption_files: number
      missing_caption_files: string[]
      missing_caption_files_truncated: boolean
    }
  }
}

export interface ConfigTemplatePlaceholder {
  key: string
  label: string
  path_field: string
  placeholder: string
}

export interface ConfigTemplate {
  id: string
  name: string
  description: string
  arch: string
  placeholders: ConfigTemplatePlaceholder[]
  config: Record<string, unknown>
}
