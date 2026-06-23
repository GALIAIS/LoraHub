import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      spellCheck={false}
      className={cn(
        "min-h-24 w-full min-w-0 resize-y rounded-[6px] border border-[var(--control-border)] bg-[var(--control-fill)] px-3 py-2 text-sm shadow-[0_1px_0_rgba(255,255,255,0.44)_inset] transition-[color,background-color,border-color,box-shadow] duration-150 outline-none placeholder:text-muted-foreground hover:border-[var(--control-border-hover)] hover:bg-[var(--control-fill-hover)] focus-visible:border-ring focus-visible:bg-background focus-visible:ring-3 focus-visible:ring-ring/35 disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 dark:shadow-[0_1px_0_rgba(255,255,255,0.04)_inset] dark:disabled:bg-input/80 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40",
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
