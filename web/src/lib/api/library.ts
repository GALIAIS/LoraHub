import { http } from "./core"

// --------------------------------------------------------------------------- //
// Image Studio Library — cross-dataset tag dictionary, trigger word index,
// prompt template store. All entries are *global* (not bound to a dataset).
// --------------------------------------------------------------------------- //

export interface LibraryTagEntry {
  tag: string
  category: string
  aliases: string[]
  color: string | null
  notes: string | null
  createdAt: string
  updatedAt: string
}

export interface LibraryTriggerEntry {
  triggerWord: string
  characterName: string | null
  concept: string | null
  datasets: string[]
  promptHint: string | null
  createdAt: string
  updatedAt: string
}

export interface LibraryPromptEntry {
  id: string
  name: string
  category: string
  body: string
  vars: string[]
  isDefault: boolean
  notes: string | null
  createdAt: string
  updatedAt: string
}

// ----- Tags ----------------------------------------------------------------

export async function libraryListTags(params: {
  category?: string
  search?: string
} = {}): Promise<{ tags: LibraryTagEntry[] }> {
  const qs = new URLSearchParams()
  if (params.category) qs.set("category", params.category)
  if (params.search) qs.set("search", params.search)
  const suffix = qs.toString() ? `?${qs.toString()}` : ""
  return http<{ tags: LibraryTagEntry[] }>(
    `/image-studio/library/tags${suffix}`,
  )
}

export async function libraryUpsertTag(body: {
  tag: string
  category?: string
  aliases?: string[]
  color?: string | null
  notes?: string | null
}): Promise<LibraryTagEntry> {
  return http<LibraryTagEntry>(
    `/image-studio/library/tags/${encodeURIComponent(body.tag)}`,
    { method: "PUT", body: JSON.stringify(body) },
  )
}

export async function libraryDeleteTag(
  tag: string,
): Promise<{ deleted: true; tag: string }> {
  return http<{ deleted: true; tag: string }>(
    `/image-studio/library/tags/${encodeURIComponent(tag)}`,
    { method: "DELETE" },
  )
}

// ----- Trigger words --------------------------------------------------------

export async function libraryListTriggers(params: {
  characterName?: string
  search?: string
} = {}): Promise<{ triggers: LibraryTriggerEntry[] }> {
  const qs = new URLSearchParams()
  if (params.characterName) qs.set("characterName", params.characterName)
  if (params.search) qs.set("search", params.search)
  const suffix = qs.toString() ? `?${qs.toString()}` : ""
  return http<{ triggers: LibraryTriggerEntry[] }>(
    `/image-studio/library/triggers${suffix}`,
  )
}

export async function libraryUpsertTrigger(body: {
  triggerWord: string
  characterName?: string | null
  concept?: string | null
  datasets?: string[]
  promptHint?: string | null
}): Promise<LibraryTriggerEntry> {
  return http<LibraryTriggerEntry>(
    `/image-studio/library/triggers/${encodeURIComponent(body.triggerWord)}`,
    { method: "PUT", body: JSON.stringify(body) },
  )
}

export async function libraryDeleteTrigger(
  triggerWord: string,
): Promise<{ deleted: true; triggerWord: string }> {
  return http<{ deleted: true; triggerWord: string }>(
    `/image-studio/library/triggers/${encodeURIComponent(triggerWord)}`,
    { method: "DELETE" },
  )
}

// ----- Prompt templates -----------------------------------------------------

export async function libraryListPrompts(params: {
  category?: string
} = {}): Promise<{ prompts: LibraryPromptEntry[] }> {
  const qs = params.category
    ? `?category=${encodeURIComponent(params.category)}`
    : ""
  return http<{ prompts: LibraryPromptEntry[] }>(
    `/image-studio/library/prompts${qs}`,
  )
}

export async function libraryCreatePrompt(body: {
  name: string
  category?: string
  body?: string
  vars?: string[]
  isDefault?: boolean
  notes?: string | null
}): Promise<LibraryPromptEntry> {
  return http<LibraryPromptEntry>("/image-studio/library/prompts", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function libraryUpsertPrompt(
  promptId: string,
  body: {
    name: string
    category?: string
    body?: string
    vars?: string[]
    isDefault?: boolean
    notes?: string | null
  },
): Promise<LibraryPromptEntry> {
  return http<LibraryPromptEntry>(
    `/image-studio/library/prompts/${encodeURIComponent(promptId)}`,
    { method: "PUT", body: JSON.stringify(body) },
  )
}

export async function libraryDeletePrompt(
  promptId: string,
): Promise<{ deleted: true; id: string }> {
  return http<{ deleted: true; id: string }>(
    `/image-studio/library/prompts/${encodeURIComponent(promptId)}`,
    { method: "DELETE" },
  )
}
