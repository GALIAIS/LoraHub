import { useState, useEffect, useRef } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { ArrowLeft, CheckCheck, Play, Save, XCircle } from "lucide-react"
import { api, type ValidationFieldError } from "@/lib/api"
import { fieldDisplay } from "@/lib/field-labels"
import { toastApiError } from "@/lib/toast-api-error"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ConfigForm, type ConfigFormValue } from "@/components/config-form"
// LLM advisor temporarily disabled for stability (see toolbar comment).
// Imports kept commented so a one-line revert restores the entry point.
// import { LlmAdvisorButton } from "@/components/config-form/llm-advisor-button"
// import { setIn as setByPath } from "@/components/config-form/types"
import type { Mode } from "../types"
import { buildDefaults } from "../utils"
import { ErrorBanner } from "./error-banner"
import { PreflightPanel } from "./preflight-panel"

export function ConfigEditor({
  mode,
  setMode,
}: {
  mode: { kind: "edit"; name: string } | { kind: "new" }
  setMode: (m: Mode) => void
}) {
  const isNew = mode.kind === "new"
  const navigate = useNavigate()
  const qc = useQueryClient()

  const sourceQuery = useQuery({
    queryKey: ["config", isNew ? "__new__" : mode.name],
    queryFn: () => (isNew ? Promise.resolve(null) : api.getConfig(mode.name)),
  })
  // Pull the workbench default backend so a fresh "新建" config opens
  // pre-wired for whatever backend the user has actually configured
  // (set in 设置 → 默认后端). Saved configs ignore this — they keep
  // whatever backend they were saved with.
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 60_000,
  })
  const defaultBackend = settingsQuery.data?.settings?.default_backend

  const [draft, setDraft] = useState<ConfigFormValue | null>(null)
  const [name, setName] = useState<string>(isNew ? "" : mode.name)
  const [errors, setErrors] = useState<ValidationFieldError[]>([])

  // Identify the (kind, name) pair the draft was last seeded against.
  // The seeding effect below only re-fires when this pair *changes* —
  // a background refetch of the same config no longer wipes a draft
  // the user is editing. Switching to a different config (or new ↔
  // edit) still rebuilds, which is what users expect.
  const seededForRef = useRef<string | null>(null)

  useEffect(() => {
    const seedKey = isNew ? "__new__" : mode.name
    const alreadySeeded = seededForRef.current === seedKey
    if (isNew) {
      // Don't seed until settings have loaded — buildDefaults reads
      // default_backend to pick the right initial arch + sub-config.
      // First-load with settings undefined falls back to kohya which
      // is fine; once settings land we still want to swap to the user
      // preference, but only on the *initial* seed for this editor.
      if (alreadySeeded) return
      setDraft(buildDefaults(defaultBackend))
      setName("")
      seededForRef.current = seedKey
    } else if (sourceQuery.data?.parsed) {
      if (alreadySeeded) return
      setDraft(sourceQuery.data.parsed as unknown as ConfigFormValue)
      setName(mode.name)
      seededForRef.current = seedKey
    }
  }, [isNew, sourceQuery.data, mode, defaultBackend])

  const validate = useMutation({
    mutationFn: () =>
      api.validateConfig((draft ?? {}) as unknown as Record<string, unknown>),
    onSuccess: (resp) => setErrors(resp.errors ?? []),
  })

  const save = useMutation({
    mutationFn: async (opts: { overwrite: boolean; thenLaunch: boolean }) => {
      if (!draft) throw new Error("当前没有草稿可保存")
      const payload = draft as unknown as Record<string, unknown>
      const v = await api.validateConfig(payload)
      if (!v.valid) {
        setErrors(v.errors ?? [])
        throw new Error("配置校验未通过")
      }
      const cleanName = name.trim()
      if (!cleanName) throw new Error("配置名称不能为空")
      const saved = await api.saveConfig(cleanName, payload, opts.overwrite || !isNew)
      qc.invalidateQueries({ queryKey: ["configs"] })
      qc.invalidateQueries({ queryKey: ["config", cleanName] })
      if (opts.thenLaunch) {
        const job = await api.createJob(payload)
        qc.invalidateQueries({ queryKey: ["jobs"] })
        navigate("/jobs")
        return { saved, job }
      }
      setMode({ kind: "preview", name: cleanName })
      return { saved }
    },
    onError: (err) => {
      toastApiError(err, { title: "保存或启动训练失败" })
    },
  })

  const loading = !isNew && sourceQuery.isLoading

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-3 shrink-0">
        <Button
          size="sm"
          variant="ghost"
          onClick={() =>
            setMode(
              isNew
                ? ({ kind: "preview", name: "" } as Mode)
                : { kind: "preview", name: mode.name },
            )
          }
        >
          <ArrowLeft className="size-3" /> 返回
        </Button>
        <div className="flex-1 min-w-0">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-1">
            {isNew ? "新建配置" : "编辑配置"}
          </div>
          {isNew ? (
            <Input
              value={name}
              placeholder="my_character"
              onChange={(e) => setName(e.target.value)}
              className="font-mono w-64 h-8"
            />
          ) : (
            <div className="text-base font-semibold tracking-tight font-mono truncate">
              {name}
            </div>
          )}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => validate.mutate()}
          disabled={!draft || validate.isPending}
        >
          <CheckCheck className="size-3" /> 校验
        </Button>
        {/*
          LLM-driven 智能推荐 entry — temporarily disabled for stability.
          The upstream proxy fronting the configured AI provider was
          dropping multi-thousand-token prompts at the 60s mark with
          ``Server disconnected``, which surfaced as a 422 in the form
          and confused users. Component + API + backend route still
          exist but the button is hidden so nothing reaches that path
          until we rework it onto the streaming code path. Re-enable
          by uncommenting once /api/configs/llm-advise stops returning
          503.
        */}
        <Button
          size="sm"
          variant="outline"
          onClick={() => save.mutate({ overwrite: !isNew, thenLaunch: false })}
          disabled={!draft || save.isPending}
        >
          <Save className="size-3" /> {save.isPending ? "保存中…" : "保存"}
        </Button>
        <Button
          size="sm"
          onClick={() => save.mutate({ overwrite: !isNew, thenLaunch: true })}
          disabled={!draft || save.isPending}
        >
          <Play className="size-3" /> 保存并训练
        </Button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-4 pb-4 pt-3 space-y-3">
          {validate.data?.valid === true && (
            <div className="rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-4 py-2 text-xs text-emerald-700 dark:text-emerald-400">
              配置校验通过。
            </div>
          )}
          {validate.data?.preflight && (
            <PreflightPanel preflight={validate.data.preflight} />
          )}
          {errors.length > 0 && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.18em] text-destructive font-semibold flex items-center gap-1.5">
                <XCircle className="size-3" /> 发现 {errors.length} 处校验错误
              </div>
              <ul className="mt-2 text-xs text-destructive space-y-1">
                {errors.slice(0, 8).map((e, i) => {
                  const raw = e.loc.join(".")
                  const fd = fieldDisplay(raw)
                  return (
                    <li key={i} className="space-y-0.5">
                      <div>
                        <span className="font-medium">{fd.label}</span>
                        {fd.hasLabel && (
                          <span className="ml-1 font-mono text-[10px] text-muted-foreground">
                            {fd.raw}
                          </span>
                        )}
                      </div>
                      <div className="text-muted-foreground/90 whitespace-pre-line font-mono">
                        {e.msg}
                      </div>
                    </li>
                  )
                })}
                {errors.length > 8 && (
                  <li className="text-muted-foreground">…还有 {errors.length - 8} 处未列出</li>
                )}
              </ul>
            </div>
          )}
          {save.isError && (
            <ErrorBanner title="保存失败" message={(save.error as Error).message} />
          )}

          {loading || !draft ? (
            <div className="text-sm text-muted-foreground px-2 py-6">加载中…</div>
          ) : (
            <ConfigForm value={draft} onChange={setDraft} errors={errors} />
          )}
        </div>
      </div>
    </div>
  )
}
