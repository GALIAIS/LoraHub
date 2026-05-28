/**
 * 标签词典面板 — 全局收藏的 tag 列表（带 category / aliases / 颜色 / 备注）。
 * 上方筛选栏 + 创建/编辑表单（行内展开），下方表格列出现有条目。
 */
import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Plus, Search, Trash2, X, Edit3, Save } from "lucide-react"
import { toast } from "sonner"
import {
  libraryDeleteTag,
  libraryListTags,
  libraryUpsertTag,
  type LibraryTagEntry,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

interface DraftState {
  tag: string
  category: string
  aliases: string
  color: string
  notes: string
}

const EMPTY_DRAFT: DraftState = {
  tag: "",
  category: "other",
  aliases: "",
  color: "",
  notes: "",
}

function entryToDraft(e: LibraryTagEntry): DraftState {
  return {
    tag: e.tag,
    category: e.category,
    aliases: e.aliases.join(", "),
    color: e.color ?? "",
    notes: e.notes ?? "",
  }
}

export function TagLibraryPanel() {
  const qc = useQueryClient()
  const [search, setSearch] = useState("")
  const [category, setCategory] = useState("")
  const [draft, setDraft] = useState<DraftState | null>(null)
  // 当 draft.tag 已存在于现有列表时是 edit 模式（PUT 走同一 path），否则 create。
  const [editingExistingTag, setEditingExistingTag] = useState<string | null>(null)

  const tagsQuery = useQuery({
    queryKey: ["library", "tags", category, search],
    queryFn: () =>
      libraryListTags({
        category: category || undefined,
        search: search || undefined,
      }),
  })

  const upsert = useMutation({
    mutationFn: libraryUpsertTag,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["library", "tags"] })
      setDraft(null)
      setEditingExistingTag(null)
      toast.success("标签已保存")
    },
    onError: (e) =>
      toast.error("保存失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const del = useMutation({
    mutationFn: libraryDeleteTag,
    onSuccess: (_, tag) => {
      qc.invalidateQueries({ queryKey: ["library", "tags"] })
      toast.success(`已删除 ${tag}`)
    },
    onError: (e) =>
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      }),
  })

  const onSubmit = () => {
    if (!draft) return
    if (!draft.tag.trim()) {
      toast.error("tag 不能为空")
      return
    }
    upsert.mutate({
      tag: draft.tag.trim(),
      category: draft.category.trim() || "other",
      aliases: draft.aliases
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      color: draft.color.trim() || null,
      notes: draft.notes.trim() || null,
    })
  }

  const onEdit = (e: LibraryTagEntry) => {
    setDraft(entryToDraft(e))
    setEditingExistingTag(e.tag)
  }

  const tags = tagsQuery.data?.tags ?? []
  const editing = editingExistingTag !== null

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Filters */}
      <div className="flex items-center gap-2 border-b px-6 py-2.5">
        <div className="relative">
          <Search className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索 tag / alias / 备注"
            className="h-8 w-64 pl-8 text-xs"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Input
          placeholder="筛选分类（character / quality / ...）"
          className="h-8 w-56 text-xs"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        />
        <div className="flex-1" />
        {!draft && (
          <Button
            size="sm"
            onClick={() => {
              setDraft({ ...EMPTY_DRAFT })
              setEditingExistingTag(null)
            }}
            className="h-8 gap-1"
          >
            <Plus className="size-3.5" />
            新建
          </Button>
        )}
      </div>

      {/* Inline draft editor */}
      {draft && (
        <div className="border-b bg-muted/30 px-6 py-3 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium">
              {editing ? `编辑 · ${editingExistingTag}` : "新建标签"}
            </span>
            <div className="flex-1" />
            <Button
              size="sm"
              variant="ghost"
              className="h-7 gap-1 text-xs"
              onClick={() => {
                setDraft(null)
                setEditingExistingTag(null)
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
            <Field label="tag">
              <Input
                value={draft.tag}
                disabled={editing}
                onChange={(e) => setDraft({ ...draft, tag: e.target.value })}
                placeholder="blue hair"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="分类">
              <Input
                value={draft.category}
                onChange={(e) =>
                  setDraft({ ...draft, category: e.target.value })
                }
                placeholder="character / quality / outfit / ..."
                className="h-8 text-xs"
              />
            </Field>
            <Field label="颜色（可选）">
              <Input
                value={draft.color}
                onChange={(e) => setDraft({ ...draft, color: e.target.value })}
                placeholder="#3b82f6"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="别名（逗号分隔）">
              <Input
                value={draft.aliases}
                onChange={(e) =>
                  setDraft({ ...draft, aliases: e.target.value })
                }
                placeholder="azure hair, sapphire hair"
                className="h-8 text-xs"
              />
            </Field>
            <Field label="备注" className="sm:col-span-2 lg:col-span-2">
              <Input
                value={draft.notes}
                onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
                placeholder="自由备注"
                className="h-8 text-xs"
              />
            </Field>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {tagsQuery.isLoading ? (
          <p className="px-6 py-8 text-center text-xs text-muted-foreground">
            加载中…
          </p>
        ) : tags.length === 0 ? (
          <p className="px-6 py-12 text-center text-xs text-muted-foreground">
            暂无标签 — 点右上角"新建"添加第一条。
          </p>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-background border-b">
              <tr className="text-left text-muted-foreground">
                <th className="px-6 py-2 font-medium">tag</th>
                <th className="px-3 py-2 font-medium">分类</th>
                <th className="px-3 py-2 font-medium">别名</th>
                <th className="px-3 py-2 font-medium">备注</th>
                <th className="px-3 py-2 font-medium w-24" />
              </tr>
            </thead>
            <tbody>
              {tags.map((t) => (
                <tr key={t.tag} className="border-b hover:bg-muted/30">
                  <td className="px-6 py-2 font-mono">
                    <span
                      className="inline-flex items-center gap-1.5"
                      style={t.color ? { color: t.color } : undefined}
                    >
                      {t.color && (
                        <span
                          className="inline-block size-2 rounded-full"
                          style={{ background: t.color }}
                        />
                      )}
                      {t.tag}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant="outline" className="text-[10px]">
                      {t.category}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {t.aliases.join(", ") || "—"}
                  </td>
                  <td className="px-3 py-2 text-muted-foreground truncate max-w-xs">
                    {t.notes || "—"}
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
                        onClick={() => del.mutate(t.tag)}
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
