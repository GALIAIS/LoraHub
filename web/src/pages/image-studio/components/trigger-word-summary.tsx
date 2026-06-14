type TriggerWordSummaryProps = {
  candidates: { trigger: string; count: number }[]
  onClose: () => void
}

export function TriggerWordSummary({
  candidates,
  onClose,
}: TriggerWordSummaryProps) {
  if (candidates.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2 bg-muted/20">
      <span className="text-xs text-muted-foreground">
        数据集触发词候选（点击复制）：
      </span>
      {candidates.map((candidate) => (
        <button
          key={candidate.trigger}
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(candidate.trigger)
          }}
          className="inline-flex items-center gap-1 rounded border bg-background px-2 py-0.5 text-[11px] font-mono hover:bg-muted transition-colors"
          title={`${candidate.count} 张图片含此触发词 · 点击复制`}
        >
          <span>{candidate.trigger}</span>
          <span className="text-muted-foreground">·{candidate.count}</span>
        </button>
      ))}
      <button
        type="button"
        onClick={onClose}
        className="ml-auto text-xs text-muted-foreground hover:text-foreground"
      >
        关闭
      </button>
    </div>
  )
}
