import { createContext, useContext, useMemo, type ReactNode } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, useSystemStream, type SystemSnapshot } from "@/lib/api"
import type { StreamStatus } from "@/lib/use-event-stream"

const POLL_INTERVAL_MS = 5_000

interface SystemTelemetryValue {
  snapshot: SystemSnapshot | null
  status: StreamStatus
  live: boolean
}

const SystemTelemetryContext = createContext<SystemTelemetryValue | null>(null)

export function SystemTelemetryProvider({ children }: { children: ReactNode }) {
  const stream = useSystemStream(true)
  const polled = useQuery({
    queryKey: ["system", "summary"],
    queryFn: api.getSystemSummary,
    refetchInterval: stream.status === "open" ? false : POLL_INTERVAL_MS,
    staleTime: 1_000,
  })
  const value = useMemo<SystemTelemetryValue>(
    () => ({
      snapshot: stream.snapshot ?? polled.data ?? null,
      status: stream.status,
      live: stream.status === "open",
    }),
    [polled.data, stream.snapshot, stream.status],
  )
  return (
    <SystemTelemetryContext.Provider value={value}>
      {children}
    </SystemTelemetryContext.Provider>
  )
}

export function useSystemTelemetry() {
  const context = useContext(SystemTelemetryContext)
  if (!context) {
    throw new Error("useSystemTelemetry must be used inside SystemTelemetryProvider")
  }
  return context
}
