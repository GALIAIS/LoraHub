/**
 * 触发词索引面板 — 维护"trigger word ↔ 角色 / 概念 ↔ 数据集"映射。
 * 跟标签词典同构，但字段是 trigger / characterName / concept / datasets / promptHint。
 */
import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Plus, Search, Trash2, X, Edit3, Save } from "lucide-react"
import { toast } from "sonner"
import {
  libraryDeleteTrigger,
  libraryListTriggers,
  libraryUpsertTrigger,
  type LibraryTriggerEntry,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

interface DraftState {
  triggerWord: string
  characterName: string
  concept: string
  datasets: string
  promptHint: string
}

const EMPTY_DRAFT: DraftState = {
  triggerWord: "",
  characterName: "",
  concept: "",
  datasets: "",
  promptHint: "",
}

function entryToDraft(e: LibraryTriggerEntry): DraftState {
  return {
    triggerWord: e.triggerWord,
    characterName: e.characterName ?? "",
    concept: e.concept ?? "",
    datasets: e.datasets.join(", "),
    promptHint: e.promptHint ?? "",
  }
}

export function TriggerLibraryPanel() {
  const qc = useQueryClient()
  const [search, setSearch] = useState("")
  const [characterName, setCharacterName] = useState("")
  const [draft, setDraft] = useState<DraftState | null>(null)
  const [editingExisting, setEditingExisting] = useState<string | null>(null)

  const triggersQuery = useQuery({
    queryKey: ["library", "triggers", characterName, search],
    queryFn: () =>
      libraryListTriggers({
        characterName: characterName || undefined,
        search: search || undefined,
      }),
  })

  const upsert = useMutation({
    mutationFn: libraryUpsertTrigger,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["library", "triggers"] })
      setDraft(null)
      setEditingExisting(null)
      toast.success("触发词已保存")
    },
    onError: (e) =>
      toast.error("保存失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const del = useMutation({
    mutationFn: libraryDeleteTrigger,
    onSuccess: (_, trigger) => {
      qc.invalidateQueries({ queryKey: ["library", "triggers"] })
      toast.success(`已删除 ${trigger}`)
    },
    onError: (e) =>
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const onSubmit = () => {
    if (!draft) return
    if (!draft.triggerWord.trim()) {
      toast.error("trigger 不能为空")
      return
    }
    upsert.mutate({
      triggerWord: draft.triggerWord.trim(),
      characterName: draft.characterName.trim() || null,
      concept: draft.concept.trim() || null,
      datasets: draft.datasets
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      promptHint: draft.promptHint.trim() || null,
    })
  }

  const onEdit = (e: LibraryTriggerEntry) => {
    setDraft(entryToDraft(e))
    setEditingExisting(e.triggerWord)
  }

  const triggers = triggersQuery.data?.triggers ?? []
  const editing = editingExisting !== null

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b px-6 py-2.5">
        <div className="relative">
          <Search className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索 trigger / 角色 / 概念"
            className="h-8 w-64 pl-8 text-xs"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Input
          placeholder="按角色名筛选"
          className="h-8 w-48 text-xs"
          value={characterName}
          onChange={(e) => setCharacterName(e.target.value)}
        />
        <div className="flex-1" />
        {!draft && (
          <Button
            size="sm"
            onClick={() => {
              setDraft({ ...EMPTY_DRAFT })
              setEditingExisting(null)
            }}
            className="h-8 gap-1"
          >
            <Plus className="size-3.5" />
            新建
          </Button>
        )}
      </div>

      {draft && (
        <div className="border-b bg-muted/30 px-6 py-3 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium">
              {editing ? `编辑 · ${editingExisting}` : "新建触发词"}
            </span>
            <div className="flex-1" />
            <Button
              size="sm"
              variant="ghost"
              className="h-7 gap-1 text-xs"
              onClick={() => {
                setDraft(null)
                setEditingExisting(null)
              }}
            >
              <X className="size-3" />
              取消
            </Button>
            <Button
              size="sm"
              className="h-7 gap-1 text-xs"
              disabled={upsert.isPending}
              onClick={onSubmit}
            >
              <Save className="size-3" />
              保存
            </Button>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <Field label="trigger word">
              <Input
                value={draft.triggerWord}
                disabled={editing}
                onChange={(e) =>
                  setDraft({ ...draft, triggerWord: e.target.value })
                }
                placeholder="aelina_v2"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="角色名">
              <Input
                value={draft.characterName}
                onChange={(e) =>
                  setDraft({ ...draft, characterName: e.target.value })
                }
                placeholder="Aelina"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="概念（风格 / 镜头 / ...）">
              <Input
                value={draft.concept}
                onChange={(e) =>
                  setDraft({ ...draft, concept: e.target.value })
                }
                placeholder="anime_style / closeup / ..."
                className="h-8 text-xs"
              />
            </Field>
            <Field label="使用过的数据集（逗号分隔）" className="sm:col-span-2">
              <Input
                value={draft.datasets}
                onChange={(e) =>
                  setDraft({ ...draft, datasets: e.target.value })
                }
                placeholder="proj-a, proj-b"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="prompt 提示" className="sm:col-span-2 lg:col-span-3">
              <Input
                value={draft.promptHint}
                onChange={(e) =>
                  setDraft({ ...draft, promptHint: e.target.value })
                }
                placeholder="placed at the front of caption"
                className="h-8 text-xs"
              />
            </Field>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        {triggersQuery.isLoading ? (
          <p className="px-6 py-8 text-center text-xs text-muted-foreground">
            加载中…
          </p>
        ) : triggers.length === 0 ? (
          <p className="px-6 py-12 text-center text-xs text-muted-foreground">
            暂无触发词。
          </p>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-background border-b">
              <tr className="text-left text-muted-foreground">
                <th className="px-6 py-2 font-medium">trigger</th>
                <th className="px-3 py-2 font-medium">角色</th>
                <th className="px-3 py-2 font-medium">概念</th>
                <th className="px-3 py-2 font-medium">数据集</th>
                <th className="px-3 py-2 font-medium w-24" />
              </tr>
            </thead>
            <tbody>
              {triggers.map((t) => (
                <tr key={t.triggerWord} className="border-b hover:bg-muted/30">
                  <td className="px-6 py-2 font-mono">{t.triggerWord}</td>
                  <td className="px-3 py-2">{t.characterName || "—"}</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {t.concept || "—"}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {t.datasets.length === 0 ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        t.datasets.map((d) => (
                          <Badge
                            key={d}
                            variant="outline"
                            className="text-[10px]"
                          >
                            {d}
                          </Badge>
                        ))
                      )}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex justify-end gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2"
                        onClick={() => onEdit(t)}
                      >
                        <Edit3 className="size-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-destructive hover:text-destructive"
                        onClick={() => del.mutate(t.triggerWord)}
                        disabled={del.isPending}
                      >
                        <Trash2 className="size-3" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
