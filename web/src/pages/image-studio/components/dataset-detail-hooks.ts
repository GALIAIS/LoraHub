import { useEffect, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { getTriggerWordsSession, imageStudioListOps } from "@/lib/api"
import { type StudioTaskRecord } from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"

export function useDatasetPendingOpsCount(path: string) {
  const opsCountQuery = useQuery({
    queryKey: ["image-studio", "ops-count", path],
    queryFn: () => imageStudioListOps(path),
    enabled: !!path,
    refetchInterval: 5000,
    select: (data) => data.ops.length,
  })

  return opsCountQuery.data ?? 0
}

export function useDatasetTaskState(path: string) {
  const queryClient = useQueryClient()
  const studioTasks = useStudioTasksFor(path)
  const [triggerWordTop, setTriggerWordTop] = useState<
    { trigger: string; count: number }[] | null
  >(null)

  const activeStudioTask: StudioTaskRecord | null =
    studioTasks.length === 0
      ? null
      : [...studioTasks].sort((a, b) => b.startedAt - a.startedAt)[0]

  const terminalSig = studioTasks
    .filter((t) => t.status !== "running")
    .map((t) => `${t.id}:${t.status}`)
    .join(",")

  useEffect(() => {
    if (terminalSig) {
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    }
  }, [terminalSig, queryClient])

  useEffect(() => {
    const completedTrigger = [...studioTasks]
      .filter((t) => t.kind === "trigger-words" && t.status === "completed")
      .sort((a, b) => b.startedAt - a.startedAt)[0]
    if (!completedTrigger) return

    let cancelled = false
    getTriggerWordsSession(completedTrigger.id)
      .then((snap) => {
        if (!cancelled) setTriggerWordTop(snap.dataset_top)
      })
      .catch(() => {
        // The task banner already reports status polling failures; this
        // summary panel is helpful but non-critical.
      })

    return () => {
      cancelled = true
    }
  }, [terminalSig, studioTasks])

  return {
    activeStudioTask,
    triggerWordTop,
    setTriggerWordTop,
  }
}
