import { useEffect, useRef, useState } from "react"
import type { ConfigFormValue } from "@/components/config-form"

// Autosave drafts of in-progress config edits to localStorage so a user
// who navigates away from /configs (or hard-refreshes) can come back to
// the same un-saved form state. Keys are namespaced by config name —
// new-config drafts share the synthetic ``__new__`` slot.
//
// Storage key shape:  ``lorahub.config-draft:<name>``  or
//                     ``lorahub.config-draft:__new__``
//
// The hook intentionally only *prompts* the user once per storageKey
// transition — re-entering the same editor without leaving the page
// does not pop the dialog again on every refetch of the source config,
// since those don't change the storage key.

const DEBOUNCE_MS = 800

function safeRead(key: string): ConfigFormValue | null {
  if (typeof window === "undefined") return null
  let raw: string | null
  try {
    raw = window.localStorage.getItem(key)
  } catch {
    return null
  }
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      // Old / corrupt entry. Drop it so we don't keep tripping over it.
      safeRemove(key)
      return null
    }
    return parsed as ConfigFormValue
  } catch {
    // Stored payload isn't valid JSON (likely a stale schema or a
    // truncated write). Silently discard — better than spamming an
    // error every time the editor mounts.
    safeRemove(key)
    return null
  }
}

function safeWrite(key: string, value: ConfigFormValue): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Quota exceeded, private mode, etc. Autosave is a courtesy, not a
    // contract — fail silently rather than disrupt the editor.
  }
}

function safeRemove(key: string): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

function serialize(value: unknown): string | null {
  try {
    return JSON.stringify(value)
  } catch {
    return null
  }
}

function isDifferent(a: unknown, b: unknown): boolean {
  const sa = serialize(a)
  const sb = serialize(b)
  if (sa === null || sb === null) return true
  return sa !== sb
}

export type DraftPersistence = {
  // Non-null when localStorage held a draft that differs from the
  // current baseline — caller should surface a restore-vs-discard
  // prompt.
  pendingRestoreDraft: ConfigFormValue | null
  // Dismiss the prompt after the caller has already applied the draft
  // back into editor state. Leaves the stored entry in place; the
  // next debounced write replaces it.
  acceptRestore: () => void
  // Dismiss the prompt and forget the stored draft.
  discardRestore: () => void
  // Clean up after a successful save — drops the stored entry.
  clearStoredDraft: () => void
}

export function useDraftPersistence(opts: {
  // localStorage namespace for the current editor target. When this
  // changes the hook re-inspects storage and (potentially) re-prompts.
  storageKey: string
  // Canonical "clean" value — server-fetched config in edit mode, or
  // the freshly-built defaults in new mode. Diffing against this is
  // how we tell whether the live ``draft`` is dirty.
  baseline: ConfigFormValue | null
  // True once ``baseline`` reflects the canonical value for
  // ``storageKey`` (settings loaded, source query landed, etc.). The
  // hook stays inert until this flips to true to avoid prompting
  // against a stale baseline from the previous editor target.
  baselineReady: boolean
  // The live editor state. Each non-null change schedules a debounced
  // write to localStorage.
  draft: ConfigFormValue | null
}): DraftPersistence {
  const { storageKey, baseline, baselineReady, draft } = opts
  const [pendingRestoreDraft, setPendingRestoreDraft] =
    useState<ConfigFormValue | null>(null)
  // Track which storageKey we already inspected so the prompt fires at
  // most once per (key, baselineReady) transition and doesn't re-pop
  // on background refetches that don't change the key.
  const inspectedKeyRef = useRef<string | null>(null)
  // Mirror prompt state into a ref so the autosave effect (which can
  // commit on the same render the inspection effect first runs) sees
  // the latest decision without a state-snapshot delay.
  const promptOpenRef = useRef(false)

  // Reset inspection bookkeeping when the storageKey changes so the
  // next baselineReady flips kick off a fresh inspection pass.
  useEffect(() => {
    inspectedKeyRef.current = null
    promptOpenRef.current = false
    setPendingRestoreDraft(null)
  }, [storageKey])

  useEffect(() => {
    if (!baselineReady) return
    if (inspectedKeyRef.current === storageKey) return
    const stored = safeRead(storageKey)
    if (stored === null) {
      inspectedKeyRef.current = storageKey
      promptOpenRef.current = false
      setPendingRestoreDraft(null)
      return
    }
    if (isDifferent(stored, baseline)) {
      promptOpenRef.current = true
      inspectedKeyRef.current = storageKey
      setPendingRestoreDraft(stored)
    } else {
      // Stored draft is identical to the current baseline — nothing
      // worth restoring. Drop the stale entry so we stay tidy.
      safeRemove(storageKey)
      inspectedKeyRef.current = storageKey
      promptOpenRef.current = false
      setPendingRestoreDraft(null)
    }
  }, [baselineReady, storageKey, baseline])

  useEffect(() => {
    if (!baselineReady) return
    // Wait until inspection has run for this key so we don't overwrite
    // a stored draft *before* the user has had a chance to decide
    // whether to restore it.
    if (inspectedKeyRef.current !== storageKey) return
    if (promptOpenRef.current) return
    if (!draft) return
    // No diff against baseline → user is in "clean" state, drop any
    // stale entry rather than persisting a redundant snapshot.
    if (!isDifferent(draft, baseline)) {
      safeRemove(storageKey)
      return
    }
    const handle = setTimeout(() => {
      safeWrite(storageKey, draft)
    }, DEBOUNCE_MS)
    return () => clearTimeout(handle)
  }, [draft, baseline, baselineReady, storageKey, pendingRestoreDraft])

  return {
    pendingRestoreDraft,
    acceptRestore: () => {
      promptOpenRef.current = false
      setPendingRestoreDraft(null)
    },
    discardRestore: () => {
      safeRemove(storageKey)
      promptOpenRef.current = false
      setPendingRestoreDraft(null)
    },
    clearStoredDraft: () => {
      safeRemove(storageKey)
      promptOpenRef.current = false
      setPendingRestoreDraft(null)
    },
  }
}
