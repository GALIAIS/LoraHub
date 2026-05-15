import { Copy, FileCheck2, FileWarning, Pencil, Trash2 } from "lucide-react"
import type { RecipeListEntry } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { RowAction } from "../types"

export function RecipeRow({
  recipe,
  active,
  onSelect,
  onAction,
}: {
  recipe: RecipeListEntry
  active: boolean
  onSelect: () => void
  onAction: (action: RowAction) => void
}) {
  return (
    <li
      onClick={onSelect}
      className={cn(
        "group/row relative px-5 py-3 cursor-pointer transition-colors",
        active
          ? "bg-accent/70 border-l-2 border-l-primary"
          : "border-l-2 border-l-transparent hover:bg-muted/40",
      )}
    >
      <div className="flex items-center gap-2 mb-1 pr-20">
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
      <div className="text-xs text-muted-foreground truncate pr-20">
        {recipe.valid ? recipe.summary : recipe.error}
      </div>
      <div
        className={cn(
          "absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5",
          "opacity-0 group-hover/row:opacity-100 focus-within:opacity-100 transition-opacity",
        )}
      >
        <RowActionButton
          label="复制"
          onClick={(e) => {
            e.stopPropagation()
            onAction("duplicate")
          }}
        >
          <Copy className="size-3" />
        </RowActionButton>
        <RowActionButton
          label="重命名"
          onClick={(e) => {
            e.stopPropagation()
            onAction("rename")
          }}
        >
          <Pencil className="size-3" />
        </RowActionButton>
        <RowActionButton
          label="删除"
          variant="destructive"
          onClick={(e) => {
            e.stopPropagation()
            onAction("delete")
          }}
        >
          <Trash2 className="size-3" />
        </RowActionButton>
      </div>
    </li>
  )
}

function RowActionButton({
  label,
  onClick,
  variant = "ghost",
  children,
}: {
  label: string
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void
  variant?: "ghost" | "destructive"
  children: React.ReactNode
}) {
  return (
    <Button
      size="icon-xs"
      variant={variant}
      aria-label={label}
      title={label}
      onClick={onClick}
    >
      {children}
    </Button>
  )
}
