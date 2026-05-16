import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Copy, Loader2, Pencil, Trash2 } from "lucide-react"
import { api, type ConfigListEntry } from "@/lib/api"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
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
  config: ConfigListEntry | null
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

export function DuplicateDialog({ open, onOpenChange, config, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open && config) {
      setNewName(`${config.name}_copy`)
      setErrorMsg(null)
    }
  }, [open, config])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!config) throw new Error("missing config")
      const trimmed = newName.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.duplicateConfig(config.name, trimmed)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["configs"] })
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
            从 <span className="font-mono">{config?.name ?? ""}</span> 创建一份副本。
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
            placeholder="my_config_copy"
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

export function RenameDialog({ open, onOpenChange, config, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open && config) {
      setNewName(config.name)
      setErrorMsg(null)
    }
  }, [open, config])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!config) throw new Error("missing config")
      const trimmed = newName.trim()
      if (!trimmed) throw new Error("名称不能为空")
      if (trimmed === config.name) throw new Error("名称未变化")
      return api.renameConfig(config.name, trimmed)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["configs"] })
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
            修改 <span className="font-mono">{config?.name ?? ""}</span> 的文件名。
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

export function DeleteDialog({ open, onOpenChange, config, onSuccess }: DialogProps) {
  const qc = useQueryClient()
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (open) setErrorMsg(null)
  }, [open])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!config) throw new Error("missing config")
      return api.deleteConfig(config.name)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["configs"] })
      onOpenChange(false)
      if (config) onSuccess?.(config.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>删除配置</AlertDialogTitle>
          <AlertDialogDescription>
            将永久删除 <span className="font-mono">{config?.filename ?? ""}</span>，无法撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <ErrorLine message={errorMsg} />
        <AlertDialogFooter>
          <AlertDialogCancel disabled={mutation.isPending}>
            取消
          </AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={mutation.isPending}
            onClick={(e) => {
              e.preventDefault()
              mutation.mutate()
            }}
          >
            {mutation.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Trash2 className="size-3" />
            )}
            {mutation.isPending ? "删除中…" : "确认删除"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
