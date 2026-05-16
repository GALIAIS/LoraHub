import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Copy, Pencil, Trash2 } from "lucide-react"
import { api, type RecipeListEntry } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type DialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  recipe: RecipeListEntry | null
  onSuccess?: (newName: string) => void
}

function ErrorLine({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
      {message}
    </div>
  )
}

export function DuplicateDialog({ open, onOpenChange, recipe, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open && recipe) {
      setNewName(`${recipe.name}_copy`)
      setErrorMsg(null)
    }
  }, [open, recipe])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!recipe) throw new Error("missing recipe")
      const trimmed = newName.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.duplicateRecipe(recipe.name, trimmed)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["recipes"] })
      onOpenChange(false)
      onSuccess?.(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>复制配置</DialogTitle>
          <DialogDescription>
            从 <span className="font-mono">{recipe?.name ?? ""}</span> 创建一份副本。
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-1.5">
          <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            新名称
          </Label>
          <Input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="font-mono"
            placeholder="my_recipe_copy"
          />
        </div>
        <ErrorLine message={errorMsg} />
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            取消
          </Button>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !newName.trim()}
          >
            <Copy className="size-3" />
            {mutation.isPending ? "复制中…" : "复制"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function RenameDialog({ open, onOpenChange, recipe, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open && recipe) {
      setNewName(recipe.name)
      setErrorMsg(null)
    }
  }, [open, recipe])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!recipe) throw new Error("missing recipe")
      const trimmed = newName.trim()
      if (!trimmed) throw new Error("名称不能为空")
      if (trimmed === recipe.name) throw new Error("名称未变化")
      return api.renameRecipe(recipe.name, trimmed)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["recipes"] })
      onOpenChange(false)
      onSuccess?.(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>重命名配置</DialogTitle>
          <DialogDescription>
            修改 <span className="font-mono">{recipe?.name ?? ""}</span> 的文件名。
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-1.5">
          <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            新名称
          </Label>
          <Input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="font-mono"
          />
        </div>
        <ErrorLine message={errorMsg} />
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            取消
          </Button>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !newName.trim()}
          >
            <Pencil className="size-3" />
            {mutation.isPending ? "重命名中…" : "重命名"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function DeleteDialog({ open, onOpenChange, recipe, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open) setErrorMsg(null)
  }, [open])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!recipe) throw new Error("missing recipe")
      return api.deleteRecipe(recipe.name)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["recipes"] })
      onOpenChange(false)
      if (recipe) onSuccess?.(recipe.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>删除配置</DialogTitle>
          <DialogDescription>
            将永久删除 <span className="font-mono">{recipe?.filename ?? ""}</span>，无法撤销。
          </DialogDescription>
        </DialogHeader>
        <ErrorLine message={errorMsg} />
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            取消
          </Button>
          <Button
            variant="destructive"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            <Trash2 className="size-3" />
            {mutation.isPending ? "删除中…" : "确认删除"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
