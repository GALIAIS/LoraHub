/**
 * Prompt 模板库面板。模板的 body 是多行的，用原生 textarea；其他字段同
 * tag/trigger 面板套路。is_default 作为复选框 — 后端会按 category 互斥。
 */
import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Plus, Trash2, X, Edit3, Save, Star } from "lucide-react"
import { toast } from "sonner"
import {
  libraryCreatePrompt,
  libraryDeletePrompt,
  libraryListPrompts,
  libraryUpsertPrompt,
  type LibraryPromptEntry,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface DraftState {
  id: string | null
  name: string
  category: string
  body: string
  vars: string
  isDefault: boolean
  notes: string
}

const EMPTY_DRAFT: DraftState = {
  id: null,
  name: "",
  category: "general",
  body: "",
  vars: "",
  isDefault: false,
  notes: "",
}

interface PromptLibraryPanelProps {
  categoryPrefix?: string
  defaultDraft?: Partial<DraftState>
}

function entryToDraft(e: LibraryPromptEntry): DraftState {
  return {
    id: e.id,
    name: e.name,
    category: e.category,
    body: e.body,
    vars: e.vars.join(", "),
    isDefault: e.isDefault,
    notes: e.notes ?? "",
  }
}

export function PromptLibraryPanel({
  categoryPrefix,
  defaultDraft,
}: PromptLibraryPanelProps = {}) {
  const qc = useQueryClient()
  const [category, setCategory] = useState(categoryPrefix ?? "")
  const [draft, setDraft] = useState<DraftState | null>(null)

  const promptsQuery = useQuery({
    queryKey: ["library", "prompts", category, categoryPrefix],
    queryFn: () =>
      categoryPrefix
        ? libraryListPrompts()
        : libraryListPrompts({ category: category || undefined }),
  })

  const create = useMutation({
    mutationFn: libraryCreatePrompt,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["library", "prompts"] })
      setDraft(null)
      toast.success("模板已创建")
    },
    onError: (e) =>
      toast.error("创建失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const upsert = useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: Parameters<typeof libraryUpsertPrompt>[1]
    }) => libraryUpsertPrompt(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["library", "prompts"] })
      setDraft(null)
      toast.success("模板已更新")
    },
    onError: (e) =>
      toast.error("更新失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const del = useMutation({
    mutationFn: libraryDeletePrompt,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["library", "prompts"] })
      toast.success("已删除模板")
    },
    onError: (e) =>
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const onSubmit = () => {
    if (!draft) return
    if (!draft.name.trim()) {
      toast.error("name 不能为空")
      return
    }
    const body = {
      name: draft.name.trim(),
      category: draft.category.trim() || "general",
      body: draft.body,
      vars: draft.vars
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      isDefault: draft.isDefault,
      notes: draft.notes.trim() || null,
    }
    if (draft.id) {
      upsert.mutate({ id: draft.id, body })
    } else {
      create.mutate(body)
    }
  }

  const prompts = (promptsQuery.data?.prompts ?? []).filter((p) =>
    categoryPrefix ? p.category.startsWith(categoryPrefix) : true,
  )
  const editing = draft?.id !== null && draft?.id !== undefined
  const submitting = create.isPending || upsert.isPending

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b px-6 py-2.5">
        {categoryPrefix ? (
          <div className="rounded border bg-muted/25 px-2 py-1 text-xs text-muted-foreground">
            {categoryPrefix}*
          </div>
        ) : (
          <Input
            placeholder="按分类筛选（caption / quality / ...）"
            className="h-8 w-64 text-xs"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          />
        )}
        <div className="flex-1" />
        {!draft && (
          <Button
            size="sm"
            onClick={() =>
              setDraft({
                ...EMPTY_DRAFT,
                ...defaultDraft,
                category:
                  defaultDraft?.category ?? categoryPrefix ?? EMPTY_DRAFT.category,
              })
            }
            className="h-8 gap-1"
          >
            <Plus className="size-3.5" />
            新建模板
          </Button>
        )}
      </div>

      {draft && (
        <div className="border-b bg-muted/30 px-6 py-3 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium">
              {editing ? `编辑模板 · ${draft.name || "未命名"}` : "新建模板"}
            </span>
            <div className="flex-1" />
            <Button
              size="sm"
              variant="ghost"
              className="h-7 gap-1 text-xs"
              onClick={() => setDraft(null)}
            >
              <X className="size-3" />
              取消
            </Button>
            <Button
              size="sm"
              className="h-7 gap-1 text-xs"
              disabled={submitting}
              onClick={onSubmit}
            >
              <Save className="size-3" />
              保存
            </Button>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="名称">
              <Input
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                placeholder="Anima Caption Default"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="分类">
              <Input
                value={draft.category}
                onChange={(e) =>
                  setDraft({ ...draft, category: e.target.value })
                }
                placeholder="caption / audit / trigger"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="变量名（逗号分隔，仅备注）">
              <Input
                value={draft.vars}
                onChange={(e) => setDraft({ ...draft, vars: e.target.value })}
                placeholder="trigger, wd14_tags"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="备注" className="sm:col-span-2 lg:col-span-3">
              <Input
                value={draft.notes}
                onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
                placeholder="使用场景 / 注意事项"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="模板正文" className="sm:col-span-2 lg:col-span-3">
              <textarea
                value={draft.body}
                onChange={(e) => setDraft({ ...draft, body: e.target.value })}
                rows={8}
                className="w-full rounded border bg-background px-2 py-1.5 text-xs font-mono outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 resize-y"
                placeholder="Write a concise booru-style caption for the image. Trigger word: {trigger}. Reference tags: {wd14_tags}."
              />
            </Field>
            <label className="flex items-center gap-2 text-xs sm:col-span-2 lg:col-span-3">
              <input
                type="checkbox"
                checked={draft.isDefault}
                onChange={(e) =>
                  setDraft({ ...draft, isDefault: e.target.checked })
                }
                className="size-3.5"
              />
              <span>设为该分类默认（保存后会取消同分类其他默认）</span>
            </label>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        {promptsQuery.isLoading ? (
          <p className="px-6 py-8 text-center text-xs text-muted-foreground">
            加载中…
          </p>
        ) : prompts.length === 0 ? (
          <p className="px-6 py-12 text-center text-xs text-muted-foreground">
            暂无 Prompt 模板。
          </p>
        ) : (
          <ul className="divide-y">
            {prompts.map((p) => (
              <li key={p.id} className="px-6 py-3 hover:bg-muted/30">
                <div className="flex items-start gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium truncate">
                        {p.name}
                      </span>
                      <Badge variant="outline" className="text-[10px]">
                        {p.category}
                      </Badge>
                      {p.isDefault && (
                        <Badge
                          className={cn(
                            "text-[10px] gap-0.5",
                            "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/40",
                          )}
                        >
                          <Star className="size-2.5 fill-current" />
                          默认
                        </Badge>
                      )}
                      {p.vars.length > 0 && (
                        <span className="text-[10px] text-muted-foreground">
                          变量: {p.vars.join(", ")}
                        </span>
                      )}
                    </div>
                    {p.notes && (
                      <p className="text-[11px] text-muted-foreground mb-1.5">
                        {p.notes}
                      </p>
                    )}
                    <pre className="text-[11px] font-mono text-muted-foreground whitespace-pre-wrap line-clamp-4 bg-muted/30 rounded px-2 py-1.5">
                      {p.body}
                    </pre>
                  </div>
                  <div className="flex gap-1 shrink-0">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2"
                      onClick={() => setDraft(entryToDraft(p))}
                    >
                      <Edit3 className="size-3" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-destructive hover:text-destructive"
                      onClick={() => del.mutate(p.id)}
                      disabled={del.isPending}
                    >
                      <Trash2 className="size-3" />
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function Field({
  label,
  children,
  className,
}: {
  label: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <label className={["flex flex-col gap-1", className].filter(Boolean).join(" ")}>
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      {children}
    </label>
  )
}
