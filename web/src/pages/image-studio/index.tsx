import { useSearchParams } from "react-router-dom"

export function ImageStudioPage() {
  const [params] = useSearchParams()
  const path = params.get("path") || ""

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Image Studio</h1>
        {path && (
          <span className="text-sm text-muted-foreground">{path}</span>
        )}
      </div>
      <div className="flex flex-1 items-center justify-center rounded-md border border-dashed p-12 text-muted-foreground">
        <p>
          Image Studio is under construction. Select a dataset folder to begin.
        </p>
      </div>
    </div>
  )
}
