import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"

export function RawConfigFallback({ content }: { content: string }) {
  return (
    <div className="rounded-[4px] border border-border bg-muted/20">
      <div className="flex items-center justify-between gap-3 border-b border-border/70 px-4 py-2">
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          原始配置
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => navigator.clipboard?.writeText(content)}
        >
          <Copy className="size-3" /> 复制
        </Button>
      </div>
      <pre className="max-h-[64vh] overflow-auto p-4 text-xs leading-5 text-foreground whitespace-pre-wrap break-words">
        {content}
      </pre>
    </div>
  )
}
