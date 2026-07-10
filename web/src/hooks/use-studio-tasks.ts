/**
 * React bindings for the global image-studio task store.
 *
 * useSyncExternalStore is the canonical React 19 way to subscribe a
 * component to an external mutable source without tearing — every
 * tick all subscribed components re-render with the same snapshot.
 *
 * Why useMemo for the dataset slice: useSyncExternalStore requires
 * its getSnapshot to return a stable reference when the store
 * hasn't changed. Filtering inline would allocate a fresh array on
 * every render and trip the "infinite loop" guard. Stashing the
 * filter inside useMemo keyed on the upstream tasks reference
 * gives us a stable identity until the store actually emits.
 */
import { useMemo, useSyncExternalStore } from "react"

import {
  getSnapshot,
  isStudioTaskActive,
  subscribe,
  type StudioTaskRecord,
} from "@/lib/studio-task-store"

const EMPTY: StudioTaskRecord[] = []

export function useStudioTasks(): StudioTaskRecord[] {
  return useSyncExternalStore(subscribe, getSnapshot, () => EMPTY)
}

export function useStudioTasksFor(
  datasetPath: string | undefined | null,
): StudioTaskRecord[] {
  const all = useStudioTasks()
  return useMemo(() => {
    if (!datasetPath) return EMPTY
    return all.filter((t) => t.datasetPath === datasetPath)
  }, [all, datasetPath])
}

export function useStudioRunningCount(): number {
  const all = useStudioTasks()
  return all.reduce((n, t) => (isStudioTaskActive(t.status) ? n + 1 : n), 0)
}
