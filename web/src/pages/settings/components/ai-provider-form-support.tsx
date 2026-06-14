import type { ReactNode } from "react"
import type { AIKeySelectionMode, AIProviderRecord } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"

export interface KeyDraftLocal {
  id: string | null
  preview: string
  value: string
  requestCount?: number
  successCount?: number
  failureCount?: number
  cooldownUntil?: string | null
  lastError?: string | null
}

export function makeDraft(provider: AIProviderRecord | null): {
  name: string
  baseUrl: string
  organization: string
  project: string
  enabled: boolean
  selectionMode: AIKeySelectionMode
  keys: KeyDraftLocal[]
  headersJson: string
} {
  return {
    name: provider?.name ?? "",
    baseUrl: provider?.baseUrl ?? "",
    organization: provider?.organization ?? "",
    project: provider?.project ?? "",
    enabled: provider?.enabled ?? true,
    selectionMode: provider?.apiKeySelectionMode ?? "round_robin",
    keys:
      provider?.apiKeys.map((key) => ({
        id: key.id,
        preview: key.preview,
        value: "",
        requestCount: key.runtime.requestCount,
        successCount: key.runtime.successCount,
        failureCount: key.runtime.failureCount,
        cooldownUntil: key.runtime.cooldownUntil,
        lastError: key.runtime.lastError,
      })) ?? [],
    headersJson:
      provider && Object.keys(provider.headers).length > 0
        ? JSON.stringify(provider.headers, null, 2)
        : "",
  }
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px]">{label}</Label>
      {children}
      {hint && (
        <p className="text-[10px] text-muted-foreground/85 leading-relaxed">
          {hint}
        </p>
      )}
    </div>
  )
}

export function KeyRuntimeBadge({ k }: { k: KeyDraftLocal }) {
  const onCooldown = !!k.cooldownUntil && new Date(k.cooldownUntil) > new Date()
  return (
    <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
      <span>请求 {k.requestCount ?? 0}</span>
      <span className="text-emerald-600 dark:text-emerald-400">
        ✓{k.successCount ?? 0}
      </span>
      <span className="text-destructive">✗{k.failureCount ?? 0}</span>
      {onCooldown && (
        <Badge variant="destructive" className="text-[9px] rounded-[2px]">
          冷却中
        </Badge>
      )}
      {k.lastError && (
        <span className="truncate text-destructive/85" title={k.lastError}>
          · {k.lastError}
        </span>
      )}
    </div>
  )
}
