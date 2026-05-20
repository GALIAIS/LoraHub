/**
 * Hook around `/api/system/version` — surfaces the latest release info
 * from React Query so badges (sidebar / topbar) and the maintenance
 * card share one fetch.
 *
 * Cache strategy:
 *  - 5 minutes ``staleTime``: matches the server-side cache TTL, so
 *    multiple components mounting on the same page coalesce into one
 *    network call.
 *  - 6 hours ``refetchInterval``: matches the server-side background
 *    poll so the UI stays in sync without us having to re-query
 *    aggressively from the client.
 *  - ``refetchOnWindowFocus`` left default so re-focus triggers a
 *    fresh check (the server returns from cache anyway when within
 *    TTL — cheap).
 */
import { useQuery } from "@tanstack/react-query"
import { api, type UpdateInfo } from "@/lib/api"

export type UpdateChannel = "main" | "tag"

const FIVE_MINUTES = 5 * 60 * 1000
const SIX_HOURS = 6 * 60 * 60 * 1000

export function useSystemVersion(
  channel: UpdateChannel = "tag",
  opts?: { enabled?: boolean },
) {
  return useQuery<UpdateInfo>({
    queryKey: ["system-version", channel],
    queryFn: () => api.getSystemVersion(channel, false),
    staleTime: FIVE_MINUTES,
    refetchInterval: SIX_HOURS,
    enabled: opts?.enabled ?? true,
  })
}

/**
 * Convenience: ``true`` when *any* configured channel reports an
 * update is available. Used by the sidebar / topbar badge so it
 * lights up regardless of which channel the user prefers.
 */
export function useHasUpdate(): boolean {
  const tag = useSystemVersion("tag")
  // We deliberately don't fetch ``main`` by default — most users want
  // tag-cut releases, not rolling-main commits. Components that want
  // to flag main-channel ahead-ness can call ``useSystemVersion("main")``
  // explicitly.
  return Boolean(tag.data?.update_available && !tag.data?.is_dirty)
}
