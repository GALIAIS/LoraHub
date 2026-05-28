/**
 * 工具库主页 — 标签词典 / 触发词 / Prompt 模板三个面板。
 *
 * 入口在 sidebar 顶部"工具库"按钮，与"全部工具"并列；URL 写 ?stage=library。
 * 工具库的三类资产是跨数据集的（不绑定 datasetPath），所以即使没选数据集也
 * 能开。Tabs 之间切换不写 URL，避免 sidebar 把"library"误判成 stage 之外的
 * 子页 — 用 ?tool= 参数控制初始 Tab 即可。
 */
import { useSearchParams } from "react-router-dom"
import { useEffect, useState } from "react"
import { BookText, Tags, Wand2 } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TagLibraryPanel } from "./tag-library-panel"
import { TriggerLibraryPanel } from "./trigger-library-panel"
import { PromptLibraryPanel } from "./prompt-library-panel"

type LibraryTab = "tags" | "triggers" | "prompts"

const TOOL_TO_TAB: Record<string, LibraryTab> = {
  "library-tags": "tags",
  "library-triggers": "triggers",
  "library-prompts": "prompts",
}

export function LibraryPage() {
  const [params] = useSearchParams()
  const initialTab: LibraryTab = (() => {
    const t = params.get("tool")
    if (t && t in TOOL_TO_TAB) return TOOL_TO_TAB[t]
    return "tags"
  })()
  const [tab, setTab] = useState<LibraryTab>(initialTab)

  // 当 URL 的 ?tool= 改变（例如从 ToolsGrid 点了不同卡片）时同步 Tab。
  useEffect(() => {
    const t = params.get("tool")
    if (t && t in TOOL_TO_TAB) setTab(TOOL_TO_TAB[t])
  }, [params])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b px-6 py-4">
        <h1 className="text-base font-semibold">工具库</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          跨数据集的标签词典、触发词索引和 Prompt 模板。新建后可在打标 / AI
          工具里直接引用。
        </p>
      </header>

      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as LibraryTab)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <div className="px-6 pt-3 pb-1 border-b border-border/60 bg-background/40">
          <TabsList variant="line">
            <TabsTrigger value="tags" className="gap-1.5">
              <Tags className="size-3.5" />
              标签词典
            </TabsTrigger>
            <TabsTrigger value="triggers" className="gap-1.5">
              <Wand2 className="size-3.5" />
              触发词索引
            </TabsTrigger>
            <TabsTrigger value="prompts" className="gap-1.5">
              <BookText className="size-3.5" />
              Prompt 模板
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="tags" className="h-full">
            <TagLibraryPanel />
          </TabsContent>
          <TabsContent value="triggers" className="h-full">
            <TriggerLibraryPanel />
          </TabsContent>
          <TabsContent value="prompts" className="h-full">
            <PromptLibraryPanel />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}
