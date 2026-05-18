/**
 * Shared `useJobsList` hook so the five pages (jobs / job-detail /
 * analysis / dashboard / gallery / global-status-bar) all funnel
 * through one observer instead of each setting its own
 * `refetchInterval` on the same `["jobs"]` key.
 *
 * Why this matters: React Query keys collide on equality, so multiple
 * `useQuery({ queryKey: ["jobs"], refetchInterval: T })` calls share
 * the cache, but the **shortest** interval wins — meaning the
 * sidebar's 5s status pill ended up being polled at 2s the whole
 * time the user was on the jobs page. With ~40 jobs in the registry
 * each fetch is non-trivial.
 *
 * Strategy:
 *   - Single 4 s base interval, considered comfortable for "near-live".
 *   - `refetchInterval` becomes a function: stops polling once the
 *     payload contains zero non-terminal jobs (everything is
 *     succeeded/failed/canceled/interrupted).
 *   - Pages re-enter polling automatically when a new job lands —
 *     the SSE / WS streams already invalidate `["jobs"]` on launch
 *     events.
 */
import { useQuery, type UseQueryOptions } from "@tanstack/react-query"
import { api, type JobSummary } from "@/lib/api"

const TERMINAL_STATES = new Set([
  "succeeded",
  "failed",
  "canceled",
  "interrupted",
])

export interface JobsListResponse {
  jobs: JobSummary[]
}

const DEFAULT_INTERVAL_MS = 4000
const STALE_MS = 1500

/**
 * Hook wrapping `GET /api/jobs`. All polling consumers should use
 * this — never set a custom `refetchInterval` on `["jobs"]`.
 */
export function useJobsList(
  options: Omit<
    UseQueryOptions<JobsListResponse, Error, JobsListResponse, ["jobs"]>,
    "queryKey" | "queryFn"
  > = {},
) {
  return useQuery({
    queryKey: ["jobs"] as const,
    queryFn: api.listJobs,
    staleTime: STALE_MS,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data || data.jobs.length === 0) return DEFAULT_INTERVAL_MS
      const hasLive = data.jobs.some((j) => !TERMINAL_STATES.has(j.state))
      return hasLive ? DEFAULT_INTERVAL_MS : false
    },
    ...options,
  })
}
