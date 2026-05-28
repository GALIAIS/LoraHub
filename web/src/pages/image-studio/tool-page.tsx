/**
 * 单个工具的独立页面容器。
 *
 * 路由 `/image-studio/tools/:toolId` 进来，根据 catalog 找到 ToolInfo,
 * 渲染 breadcrumb / dataset selector / 内容。具体内容由 tool-registry
 * 里的组件产生 — 容器负责给一个统一的外壳 + URL 同步。
 *
 * 数据集走 ?path= URL 参数，跨页保持。无数据集 + requiresDataset 时显示
 * 选择数据集的提示，避免下游 panel 跑崩。
 */
import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom"
import { ChevronLeft, FolderOpen } from "lucide-react"
import { datasetList } from "@/lib/api"
import { TOOLS, TOOL_CATEGORIES } from "./tools-catalog"
import { TOOL_COMPONENTS } from "./tool-registry"

export function ToolPage() {
  const { toolId = "" } = useParams<{ toolId: string }>()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const datasetPath = params.get("path") || ""

  const tool = TOOLS.find((t) => t.id === toolId)
  if (!tool) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="text-center">
          <p className="text-sm font-medium">未知工具</p>
          <p className="mt-1 text-xs text-muted-foreground">
            找不到 id 为 <code className="font-mono">{toolId}</code> 的工具。
          </p>
          <button
            type="button"
            onClick={() => navigate("/image-studio")}
            className="mt-4 rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground"
          >
            回到工具广场
          </button>
        </div>
      </div>
    )
  }

  const category = TOOL_CATEGORIES.find((c) => c.id === tool.category)
  const Body = TOOL_COMPONENTS[tool.id]
  const Icon = tool.icon
  const needsDataset = tool.requiresDataset && !datasetPath

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="border-b px-6 py-3">
        <nav className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Link to="/image-studio" className="hover:text-foreground">
            工具广场
          </Link>
          <span>/</span>
          <span>{category?.label ?? tool.category}</span>
          <span>/</span>
          <span className="text-foreground">{tool.label}</span>
        </nav>
        <div className="mt-1.5 flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate("/image-studio")}
            className="rounded p-1 hover:bg-accent"
            title="回到工具广场"
          >
            <ChevronLeft className="size-3.5" />
          </button>
          <Icon className="size-4 text-muted-foreground" />
          <h1 className="text-sm font-semibold">{tool.label}</h1>
          {tool.async && (
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-700 dark:text-amber-400">
              异步
            </span>
          )}
          {tool.writes && (
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-700 dark:text-emerald-400">
              写入
            </span>
          )}
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{tool.description}</p>
        {tool.requiresDataset && (
          <DatasetSelectorRow
            datasetPath={datasetPath}
            onChange={(p) => {
              const next = new URLSearchParams(params)
              if (p) next.set("path", p)
              else next.delete("path")
              setParams(next, { replace: true })
            }}
          />
        )}
      </header>

      <div className="flex-1 min-h-0 overflow-hidden">
        {needsDataset ? (
          <NeedsDatasetEmpty />
        ) : Body ? (
          <Body datasetPath={datasetPath} />
        ) : (
          <NotImplementedEmpty toolLabel={tool.label} />
        )}
      </div>
    </div>
  )
}

function DatasetSelectorRow({
  datasetPath,
  onChange,
}: {
  datasetPath: string
  onChange: (path: string) => void
}) {
  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: datasetList,
  })
  const datasets = datasetsQuery.data?.datasets ?? []
  return (
    <label className="mt-2 flex items-center gap-2">
      <FolderOpen className="size-3.5 text-muted-foreground" />
      <span className="text-[11px] text-muted-foreground">数据集</span>
      <select
        value={datasetPath}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border bg-background px-2 py-1 text-xs"
      >
        <option value="">— 选择数据集 —</option>
        {datasets.map((d) => (
          <option key={d.name} value={d.path}>
            {d.name} ({d.imageCount})
          </option>
        ))}
      </select>
    </label>
  )
}

function NeedsDatasetEmpty() {
  return (
    <div className="flex h-full items-center justify-center p-6 text-center">
      <div>
        <p className="text-sm font-medium">需要先选择一个数据集</p>
        <p className="mt-1 text-xs text-muted-foreground">
          从上方的数据集下拉框里选一个，或回到工具广场创建新数据集。
        </p>
      </div>
    </div>
  )
}

function NotImplementedEmpty({ toolLabel }: { toolLabel: string }) {
  return (
    <div className="flex h-full items-center justify-center p-6 text-center">
      <div>
        <p className="text-sm font-medium">{toolLabel} · 即将上线</p>
        <p className="mt-1 text-xs text-muted-foreground">
          这个工具的页面还在建设中，先在这里占位。
        </p>
      </div>
    </div>
  )
}
