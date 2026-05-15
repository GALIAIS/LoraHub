import { useEffect, useRef, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Upload } from "lucide-react"
import { api } from "@/lib/api"
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
import { Switch } from "@/components/ui/switch"

/** Strip ".yaml" / ".yml" off a file's basename to seed the name input. */
function stemOf(filename: string): string {
  const idx = filename.lastIndexOf(".")
  if (idx <= 0) return filename
  const ext = filename.slice(idx + 1).toLowerCase()
  if (ext === "yaml" || ext === "yml") return filename.slice(0, idx)
  return filename
}

export function ImportDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImported: (name: string) => void
}) {
  const qc = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState("")
  const [overwrite, setOverwrite] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      setFile(null)
      setName("")
      setOverwrite(false)
      setErrorMsg(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }, [open])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("请选择 YAML 文件")
      const trimmed = name.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.importRecipe(trimmed, file, overwrite)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["recipes"] })
      onOpenChange(false)
      onImported(resp.name)
    },
    onError: (err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>导入配方</DialogTitle>
          <DialogDescription>
            上传一个 YAML 文件并保存到本地配方目录。文件大小上限 1 MiB。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="flex flex-col gap-1.5">
            <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              YAML 文件
            </Label>
            <Input
              ref={fileInputRef}
              type="file"
              accept=".yaml,.yml"
              onChange={(e) => {
                const next = e.target.files?.[0] ?? null
                setFile(next)
                setErrorMsg(null)
                if (next && !name.trim()) {
                  setName(stemOf(next.name))
                }
              }}
              className="font-mono text-xs"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              保存为
            </Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="font-mono"
              placeholder="my_recipe"
            />
          </div>

          <label className="inline-flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none">
            <Switch
              size="sm"
              checked={overwrite}
              onCheckedChange={(v) => setOverwrite(Boolean(v))}
            />
            同名时覆盖
          </label>
        </div>

        {errorMsg && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMsg}
          </div>
        )}

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
            disabled={!file || !name.trim() || mutation.isPending}
          >
            <Upload className="size-3" />
            {mutation.isPending ? "导入中…" : "导入"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
