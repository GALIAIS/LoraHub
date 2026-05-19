import { Loader2 } from "lucide-react"

import { cn } from "@/lib/utils"

// Minimal shadcn-style spinner. Wraps lucide's ``Loader2`` with a
// constant rotation animation; consumers control size via ``className``
// (defaults to the same ``size-4`` baseline as Button's auto-sized
// icons). ``role`` + ``aria-label`` are emitted so screen readers
// announce the loading state when a button uses the spinner mid-action.
function Spinner({
  className,
  ...props
}: React.SVGProps<SVGSVGElement>) {
  return (
    <Loader2
      role="status"
      aria-label="加载中"
      className={cn("animate-spin", className)}
      {...props}
    />
  )
}

export { Spinner }
