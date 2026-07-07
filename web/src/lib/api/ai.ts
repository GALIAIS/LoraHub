// AI subsystem (ShiroManager-shaped) — providers, models, routes, and
// task invocations. The fetcher methods themselves live on `api.*` in
// `./client`; this module owns the type surface plus the task-id list.

export type AIReasoningEffort = "low" | "medium" | "high"
export type AIKeySelectionMode = "round_robin" | "random"
export type AIModelSource = "manual" | "discovered"

export const AI_TASK_IDS = [
  "global.default",
  "tagging.assist",
  "caption.rewrite",
  "dataset.analyze",
  "training.diagnose",
  "error.diagnose",
  "quality.score",
  "trigger.suggest",
  "config.recommend",
] as const
export type AITaskId = (typeof AI_TASK_IDS)[number]

export interface AIProviderKeyRuntime {
  requestCount: number
  successCount: number
  failureCount: number
  consecutiveFailures: number
  lastUsedAt: string | null
  lastSucceededAt: string | null
  lastFailedAt: string | null
  lastError: string | null
  cooldownUntil: string | null
}

export interface AIProviderKeyRecord {
  id: string
  preview: string
  createdAt: string
  updatedAt: string
  runtime: AIProviderKeyRuntime
}

export interface AIProviderKeyDraft {
  id?: string | null
  value?: string
  preview?: string
}

export interface AIProviderRecord {
  id: string
  name: string
  kind: "openai-compatible"
  baseUrl: string
  organization: string
  project: string
  headers: Record<string, string>
  enabled: boolean
  hasApiKey: boolean
  apiKeyPreview: string
  apiKeyCount: number
  apiKeySelectionMode: AIKeySelectionMode
  apiKeys: AIProviderKeyRecord[]
  createdAt: string
  updatedAt: string
}

export interface AIProviderDraft {
  id?: string | null
  name: string
  kind?: "openai-compatible"
  baseUrl?: string
  organization?: string
  project?: string
  headers?: Record<string, string>
  enabled?: boolean
  apiKeySelectionMode?: AIKeySelectionMode
  apiKeys?: AIProviderKeyDraft[]
  apiKey?: string
  clearApiKey?: boolean
}

export interface AIModelRecord {
  id: string
  providerId: string
  modelId: string
  displayName: string
  source: AIModelSource
  enabled: boolean
  raw: Record<string, unknown>
  createdAt: string
  updatedAt: string
}

export interface AIModelDraft {
  id?: string | null
  providerId: string
  modelId: string
  displayName: string
  source?: AIModelSource
  enabled?: boolean
  raw?: Record<string, unknown>
}

export interface AIRouteRecord {
  taskId: string
  providerId: string | null
  modelId: string | null
  systemPrompt: string
  stream: boolean | null
  temperature: number | null
  topP: number | null
  frequencyPenalty: number | null
  presencePenalty: number | null
  maxOutputTokens: number | null
  seed: number | null
  reasoningEffort: AIReasoningEffort | null
  thinkingBudgetTokens: number | null
  includeReasoning: boolean | null
  stopSequences: string[]
  extraBodyJson: string
  enabled: boolean
  createdAt: string
  updatedAt: string
}

export interface AIRouteDraft {
  taskId: string
  providerId?: string | null
  modelId?: string | null
  systemPrompt?: string
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[]
  extraBodyJson?: string
  enabled?: boolean
}

export interface AIConnectionTestInput {
  providerId: string
  modelId?: string | null
  prompt?: string | null
  systemPrompt?: string | null
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[] | null
  extraBodyJson?: string | null
}

export interface AIInvokeTaskInput {
  taskId: string
  prompt: string
  systemPrompt?: string | null
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[] | null
  extraBodyJson?: string | null
}

export interface AIUsage {
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
}

export interface AIInvokeTaskResult {
  taskId: string
  providerId: string
  providerName: string
  modelId: string
  content: string
  reasoning: string | null
  finishReason: string | null
  usage: AIUsage | null
}

export interface AIConnectionTestResult {
  ok: boolean
  providerId: string
  providerName: string
  modelCount: number
  models: Array<{
    id: string | null
    object: string | null
    ownedBy: string | null
  }>
  completion: AIInvokeTaskResult | null
  error: string | null
}
