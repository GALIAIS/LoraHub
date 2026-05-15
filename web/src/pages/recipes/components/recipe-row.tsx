import { FileCheck2, FileWarning } from "lucide-react"
import type { RecipeListEntry } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export function RecipeRow({
  recipe,
  active,
  onSelect,
}: {
  recipe: RecipeListEntry
  active: boolean
  onSelect: () => void
}) {
  return (
    <li
      onClick={onSelect}
      className={cn(
        "px-5 py-3 cursor-pointer transition-colors",
        active
          ? "bg-accent/70 border-l-2 border-l-primary"
          : "border-l-2 border-l-transparent hover:bg-muted/40",
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        {recipe.valid ? (
          <FileCheck2 className="size-3.5 text-emerald-600 dark:text-emerald-400" />
        ) : (
          <FileWarning className="size-3.5 text-destructive" />
        )}
        <span className="text-sm font-medium truncate">{recipe.name}</span>
        {recipe.arch && (
          <Badge variant="outline" className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]">
            {recipe.arch}
          </Badge>
        )}
      </div>
      <div className="text-xs text-muted-foreground truncate">
        {recipe.valid ? recipe.summary : recipe.error}
      </div>
    </li>
  )
}
