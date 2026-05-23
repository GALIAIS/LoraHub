import { useEffect, useRef, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, Upload } from "lucide-react"
import { ApiError, api, type ImportErrorDetail } from "@/lib/api"
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
  // When the server returns a structured import-error detail (yaml
  // parse failure with line / hint / etc.) we render it as a richer
  // panel below the bare error text. Null = render the legacy single
  // line view.
  const [errorDetail, setErrorDetail] = useState<ImportErrorDetail | null>(null)

  useEffect(() => {
    if (!open) {
      setFile(null)
      setName("")
      setOverwrite(false)
      setErrorMsg(null)
      setErrorDetail(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }, [open])

  const mutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("请选择 YAML 文件")
      const trimmed = name.trim()
      if (!trimmed) throw new Error("名称不能为空")
      return api.importConfig(trimmed, file, overwrite)
    },
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["configs"] })
      onOpenChange(false)
      onImported(resp.name)
    },
    onError: (err) => {
      if (err instanceof ApiError) {
        const detail = err.importErrorDetail
        if (detail) {
          setErrorDetail(detail)
          // Use the localized hint as the headline when present; the
          // raw message becomes secondary detail rendered next to a
          // line snippet.
          setErrorMsg(detail.hint ?? detail.message ?? err.message)
          return
        }
      }
      setErrorDetail(null)
      setErrorMsg(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>导入配置</DialogTitle>
          <DialogDescription>
            上传一个 YAML 文件并保存到本地配置目录。文件大小上限 1 MiB。
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
              placeholder="my_config"
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
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive whitespace-pre-wrap break-words space-y-2">
            <div className="flex items-start gap-1.5">
              <AlertTriangle className="size-3.5 mt-[1px] shrink-0" />
              <span className="font-medium leading-snug">{errorMsg}</span>
            </div>
            {errorDetail?.snippet && (
              <div className="space-y-1 pl-5">
                <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/80">
                  第 {errorDetail.line ?? "?"} 行
                  {typeof errorDetail.column === "number"
                    ? ` 第 ${errorDetail.column} 列`
                    : ""}
                </div>
                <pre className="font-mono text-[11px] leading-relaxed bg-muted/40 border border-border/60 rounded-[3px] px-2 py-1 overflow-x-auto text-foreground/85">
                  {errorDetail.snippet}
                </pre>
              </div>
            )}
            {errorDetail?.message && errorDetail.hint && (
              <details className="pl-5 text-muted-foreground">
                <summary className="cursor-pointer select-none text-[11px] hover:text-foreground">
                  查看原始 YAML 解析器消息
                </summary>
                <pre className="font-mono text-[10.5px] leading-relaxed mt-1 whitespace-pre-wrap break-words">
                  {errorDetail.message}
                </pre>
              </details>
            )}
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
