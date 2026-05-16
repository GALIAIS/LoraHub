export function ErrorBanner({ title, message }: { title: string; message: string }) {
  return (
    <div className="mx-4 mt-4 rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.18em] text-destructive font-semibold">
        {title}
      </div>
      <div className="mt-1 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
        {message}
      </div>
    </div>
  )
}
