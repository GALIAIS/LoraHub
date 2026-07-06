import { http } from "./core"
import type { SettingsState } from "./settings"

// ----------------------------------------------------------------------- //
// Error reports — Settings → 错误上报 panel + global reporter callers.
//
// Keep this surface separate from ``api`` so the panel page can pull it
// in without forcing the rest of the file into its bundle. The runtime
// reporter (``lib/error-reporter.ts``) talks to the same endpoint via
// fetch() directly to stay dependency-free.
// ----------------------------------------------------------------------- //
export interface ErrorReportItem {
  id: string
  timestamp: string
  severity: "info" | "warn" | "error" | "fatal"
  source: string
  category: string
  title: string
  message: string
  stack: string | null
  context: Record<string, unknown>
  job_id: string | null
  request_id: string | null
  request_path: string | null
  version: string
  platform: string
  fingerprint: string | null
  upstream_status: string | null
  upstream_url: string | null
  upstream_id: string | null
  upstream_error: string | null
  sent_at: string | null
  resolution_status: "open" | "resolved" | "ignored"
  resolved_at: string | null
  resolution_note: string | null
}

export interface ErrorReportListResponse {
  items: ErrorReportItem[]
  total: number
  limit: number
  offset: number
}

export interface ErrorReportSummaryResponse {
  total: number
  by_severity: Partial<Record<ErrorReportItem["severity"], number>>
  by_source: Record<string, number>
  by_resolution: Partial<Record<ErrorReportItem["resolution_status"], number>>
  upstream_attention: number
  duplicate_groups: {
    fingerprint: string
    count: number
    latest_title: string
    latest_timestamp: string
    severity: ErrorReportItem["severity"]
  }[]
}

export interface UpstreamSendResponse {
  ok: boolean
  status: string
  url: string | null
  upstream_id: string | null
  error: string | null
}

export interface UpstreamHealthResponse {
  ok: boolean
  channel: string
  url: string | null
  error: string | null
}

export interface UpstreamPreviewResponse {
  fingerprint: string
  body: ErrorReportItem
}

export const errorReportsApi = {
  list: (params: {
    limit?: number
    offset?: number
    severity?: ErrorReportItem["severity"]
    source?: string
    job_id?: string
    fingerprint?: string
    resolution_status?: ErrorReportItem["resolution_status"]
    q?: string
  } = {}) => {
    const qs = new URLSearchParams()
    if (params.limit != null) qs.set("limit", String(params.limit))
    if (params.offset != null) qs.set("offset", String(params.offset))
    if (params.severity) qs.set("severity", params.severity)
    if (params.source) qs.set("source", params.source)
    if (params.job_id) qs.set("job_id", params.job_id)
    if (params.fingerprint) qs.set("fingerprint", params.fingerprint)
    if (params.resolution_status) qs.set("resolution_status", params.resolution_status)
    if (params.q) qs.set("q", params.q)
    return http<ErrorReportListResponse>(
      `/error-reports${qs.size ? `?${qs.toString()}` : ""}`,
    )
  },
  summary: (params: {
    severity?: ErrorReportItem["severity"]
    source?: string
    job_id?: string
    fingerprint?: string
    resolution_status?: ErrorReportItem["resolution_status"]
    q?: string
  } = {}) => {
    const qs = new URLSearchParams()
    if (params.severity) qs.set("severity", params.severity)
    if (params.source) qs.set("source", params.source)
    if (params.job_id) qs.set("job_id", params.job_id)
    if (params.fingerprint) qs.set("fingerprint", params.fingerprint)
    if (params.resolution_status) qs.set("resolution_status", params.resolution_status)
    if (params.q) qs.set("q", params.q)
    return http<ErrorReportSummaryResponse>(
      `/error-reports/summary${qs.size ? `?${qs.toString()}` : ""}`,
    )
  },
  get: (id: string) =>
    http<ErrorReportItem>(`/error-reports/${encodeURIComponent(id)}`),
  delete: (id: string) =>
    http<void>(`/error-reports/${encodeURIComponent(id)}`, { method: "DELETE" }),
  setResolution: (
    id: string,
    body: { status: ErrorReportItem["resolution_status"]; note?: string | null },
  ) =>
    http<ErrorReportItem>(`/error-reports/${encodeURIComponent(id)}/resolution`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  clear: () =>
    http<{ deleted: number }>(`/error-reports/clear`, { method: "POST" }),
  exportUrl: (): string => "/api/error-reports/export",
  sendNow: (id: string) =>
    http<UpstreamSendResponse>(
      `/error-reports/${encodeURIComponent(id)}/send`,
      { method: "POST" },
    ),
  upstreamHealth: (
    draft?: {
      channel?: SettingsState["error_upstream_channel"]
      gitlab_base_url?: string
      gitlab_repo?: string
      gitlab_token?: string
      webhook_url?: string
      webhook_auth_header?: string
    },
  ) =>
    http<UpstreamHealthResponse>(`/error-reports/upstream/health`, {
      method: "POST",
      body: draft ? JSON.stringify(draft) : undefined,
      headers: draft ? { "Content-Type": "application/json" } : {},
    }),
  upstreamPreview: (id: string) =>
    http<UpstreamPreviewResponse>(
      `/error-reports/${encodeURIComponent(id)}/upstream-preview`,
    ),
}
