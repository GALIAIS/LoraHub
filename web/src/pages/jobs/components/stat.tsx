export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-card/70 px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.2em] text-muted-foreground/70">
        {label}
      </div>
      <div className="text-sm font-semibold tabular-nums truncate">{value}</div>
    </div>
  )
}
