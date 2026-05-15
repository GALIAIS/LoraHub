import { useState, useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { ArrowLeft, CheckCheck, Play, Save, XCircle } from "lucide-react"
import { api, type ValidationFieldError } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { RecipeForm, type RecipeFormValue } from "@/components/recipe-form"
import type { Mode } from "../types"
import { buildDefaults } from "../utils"
import { ErrorBanner } from "./error-banner"
import { PreflightPanel } from "./preflight-panel"

export function RecipeEditor({
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
    queryKey: ["recipe", isNew ? "__new__" : mode.name],
    queryFn: () => (isNew ? Promise.resolve(null) : api.getRecipe(mode.name)),
  })

  const [draft, setDraft] = useState<RecipeFormValue | null>(null)
  const [name, setName] = useState<string>(isNew ? "" : mode.name)
  const [errors, setErrors] = useState<ValidationFieldError[]>([])

  // Seed the draft from the source recipe (or sane defaults for new).
  useEffect(() => {
    if (isNew) {
      setDraft(buildDefaults())
      setName("")
    } else if (sourceQuery.data?.parsed) {
      setDraft(sourceQuery.data.parsed as unknown as RecipeFormValue)
      setName(mode.name)
    }
  }, [isNew, sourceQuery.data, mode])

  const validate = useMutation({
    mutationFn: () =>
      api.validateRecipe((draft ?? {}) as unknown as Record<string, unknown>),
    onSuccess: (resp) => setErrors(resp.errors ?? []),
  })

  const save = useMutation({
    mutationFn: async (opts: { overwrite: boolean; thenLaunch: boolean }) => {
      if (!draft) throw new Error("当前没有草稿可保存")
      const payload = draft as unknown as Record<string, unknown>
      const v = await api.validateRecipe(payload)
      if (!v.valid) {
        setErrors(v.errors ?? [])
        throw new Error("配方校验未通过")
      }
      const cleanName = name.trim()
      if (!cleanName) throw new Error("配方名称不能为空")
      const saved = await api.saveRecipe(cleanName, payload, opts.overwrite || !isNew)
      qc.invalidateQueries({ queryKey: ["recipes"] })
      qc.invalidateQueries({ queryKey: ["recipe", cleanName] })
      if (opts.thenLaunch) {
        const job = await api.createJob(payload)
        qc.invalidateQueries({ queryKey: ["jobs"] })
        navigate("/jobs")
        return { saved, job }
      }
      setMode({ kind: "preview", name: cleanName })
      return { saved }
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
            {isNew ? "新建配方" : "编辑配方"}
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
              配方校验通过。
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
              <ul className="mt-2 text-xs font-mono text-destructive space-y-0.5">
                {errors.slice(0, 8).map((e, i) => (
                  <li key={i}>
                    <span className="text-muted-foreground">{e.loc.join(".")}</span>: {e.msg}
                  </li>
                ))}
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
            <RecipeForm value={draft} onChange={setDraft} errors={errors} />
          )}
        </div>
      </div>
    </div>
  )
}
