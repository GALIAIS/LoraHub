import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { FilterX, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { imageStudioCaptionsBlacklist } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"

// --------------------------------------------------------------------------- //
// captions-blacklist — 输入一组 tag，一键全删
// --------------------------------------------------------------------------- //

export function CaptionsBlacklistTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [text, setText] = useState("")
  const [caseSensitive, setCaseSensitive] = useState(false)
  const [recursive, setRecursive] = useState(true)

  const tags = text
    .split(/[\n,，]/)
    .map((t) => t.trim())
    .filter(Boolean)

  const mutation = useMutation({
    mutationFn: () =>
      imageStudioCaptionsBlacklist({
        dataset_path: datasetPath,
        tags,
        case_sensitive: caseSensitive,
        recursive,
      }),
    onSuccess: (data) => {
      toast.success(
        `已删除 ${data.removed_count} 处 tag（${data.edited_count} 个文件）`,
        { description: `黑名单：${data.blacklisted_tags.join(", ")}` },
      )
      setText("")
      qc.invalidateQueries({ queryKey: ["image-studio-captions-vocab", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio-audit-report", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err) =>
      toast.error("黑名单删除失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  return (
    <div className="h-full overflow-y-auto p-4 max-w-xl">
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <FilterX className="size-3.5" />
          <span className="text-xs font-medium">标签黑名单</span>
          {tags.length > 0 && (
            <span className="ml-auto text-[11px] text-muted-foreground tabular-nums">
              {tags.length} 个 tag
            </span>
          )}
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            把要从全部 caption 删掉的 tag 一行一个粘进来（或用逗号分隔）。
            写入前自动备份原文件到 <code>.workbench/backups/</code>。
            想看现成 tag 列表，去「标签词频」工具勾选删除。
          </p>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={"masterpiece\nhigh quality\n@charA"}
            className="w-full h-32 rounded border bg-background px-2 py-1.5 text-xs font-mono resize-y"
          />
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={caseSensitive} onCheckedChange={setCaseSensitive} />
              大小写敏感
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button
            size="sm"
            disabled={tags.length === 0 || mutation.isPending}
            onClick={() => {
              if (
                !window.confirm(
                  `确认从全部 caption 中删除这 ${tags.length} 个 tag？\n（原文件备份到 .workbench/backups/，可恢复）`,
                )
              )
                return
              mutation.mutate()
            }}
            className="w-full gap-1"
          >
            {mutation.isPending ? <Loader2 className="size-3 animate-spin" /> : null}
            从所有 caption 删除
          </Button>
        </div>
      </section>
    </div>
  )
}
