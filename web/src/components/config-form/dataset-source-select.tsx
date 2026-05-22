/**
 * Dataset source picker.
 *
 * Replaces the raw PathInput field on dataset.source. Default mode:
 * dropdown of every dataset registered under datasets/ (via the
 * imageStudio listing endpoint). "高级" toggle reveals a free-form
 * PathInput so users can still point at a path that lives outside
 * the registered datasets root — needed for backward compat with
 * any yaml that already has a custom path baked in.
 */
import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Pencil } from "lucide-react"
import { datasetList } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { PathInput } from "./widgets"

interface Props {
  value: string | undefined
  onChange: (next: string) => void
  placeholder?: string
}

export function DatasetSourceSelect({ value, onChange, placeholder }: Props) {
  const datasets = useQuery({
    queryKey: ["image-studio-datasets"],
    queryFn: () => datasetList(),
  })
  const list = datasets.data?.datasets ?? []

  // Auto-flip into advanced mode when the existing value isn't one of
  // the registered datasets (typical for yaml saved before this
  // dropdown landed). Re-evaluated on every render so a manual flip
  // back into dropdown mode after fixing the path works as expected.
  const valueIsRegistered = list.some((d) => d.path === value)
  const [advanced, setAdvanced] = useState<boolean>(
    () => Boolean(value) && !valueIsRegistered,
  )

  return (
    <div className="flex flex-col gap-1.5 w-full">
      <div className="flex items-center gap-2">
        {advanced ? (
          <PathInput
            value={value ?? ""}
            onChange={onChange}
            placeholder={placeholder ?? "./datasets/my_character"}
          />
        ) : (
          <Select
            value={valueIsRegistered ? value : ""}
            onValueChange={(v) => {
              if (v) onChange(v)
            }}
          >
            <SelectTrigger className="font-mono flex-1 min-w-[14rem] h-8 text-xs">
              <SelectValue
                placeholder={
                  datasets.isLoading
                    ? "加载数据集..."
                    : list.length === 0
                      ? "datasets/ 下尚无数据集"
                      : "选择数据集..."
                }
              />
            </SelectTrigger>
            <SelectContent>
              {list.map((d) => (
                <SelectItem key={d.path} value={d.path}>
                  {d.name}{" "}
                  <span className="text-muted-foreground text-[10px] ml-2">
                    ({d.imageCount} 张)
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-[11px] gap-1 shrink-0"
          onClick={() => setAdvanced((v) => !v)}
          title={advanced ? "切回下拉" : "高级 (任意路径)"}
        >
          <Pencil className="size-3" />
          {advanced ? "下拉" : "高级"}
        </Button>
      </div>
      {!advanced && value && !valueIsRegistered && (
        <span className="text-[10px] text-amber-700 dark:text-amber-400">
          当前值不在 datasets/ 列表中,切到高级模式可查看 / 修改。
        </span>
      )}
    </div>
  )
}
