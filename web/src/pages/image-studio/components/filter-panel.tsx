import type { FilterState } from "./types"

interface FilterPanelProps {
  filters: FilterState
  onChange: (filters: FilterState) => void
  onClose: () => void
}

export function FilterPanel({ filters, onChange, onClose }: FilterPanelProps) {
  const set = <K extends keyof FilterState>(key: K, value: FilterState[K]) =>
    onChange({ ...filters, [key]: value })

  return (
    <aside className="shiro-page-aside w-52 shrink-0 overflow-y-auto p-3">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold">筛选</span>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-0.5 text-muted-foreground hover:bg-muted text-xs"
        >
          &times;
        </button>
      </div>

      <FilterGroup label="描述覆盖">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "has", label: "有描述" },
            { value: "missing", label: "缺描述" },
          ]}
          value={filters.caption}
          onChange={(v) => set("caption", v as FilterState["caption"])}
        />
      </FilterGroup>

      <FilterGroup label="质量评级">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "good", label: "优" },
            { value: "medium", label: "中" },
            { value: "bad", label: "差" },
            { value: "unrated", label: "未评" },
            { value: "favorite", label: "收藏" },
          ]}
          value={filters.quality}
          onChange={(v) => set("quality", v as FilterState["quality"])}
        />
      </FilterGroup>

      <FilterGroup label="宽高比">
        <FilterRadio
          options={[
            { value: "all", label: "全部" },
            { value: "landscape", label: "横向" },
            { value: "portrait", label: "纵向" },
            { value: "square", label: "方形" },
          ]}
          value={filters.aspect}
          onChange={(v) => set("aspect", v as FilterState["aspect"])}
        />
      </FilterGroup>
    </aside>
  )
}

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      <div className="mt-1">{children}</div>
    </div>
  )
}

function FilterRadio({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`rounded px-1.5 py-0.5 text-[11px] transition-colors ${
            value === opt.value
              ? "bg-primary text-primary-foreground"
              : "bg-muted/50 text-muted-foreground hover:bg-muted"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
