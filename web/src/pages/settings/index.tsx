import { useCallback } from "react"
import { useUrlState } from "@/lib/url-state"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { OverviewTab } from "./components/overview-tab"
import { EnvironmentTab } from "./components/environment-tab"
import { ErrorsTab } from "./components/errors-tab"
import { TaggingTab } from "./components/tagging-tab"
import { NetworkTab } from "./components/network-tab"
import { ModelsTab } from "./components/models-tab"
import { AIProvidersTab } from "./components/ai-providers-tab"
import { MaintenanceTab } from "./components/maintenance-tab"

const TAB_KEYS = [
  "overview",
  "environment",
  "network",
  "models",
  "tagging",
  "ai",
  "errors",
  "maintenance",
] as const

type TabKey = (typeof TAB_KEYS)[number]

const VALID_TABS = new Set<string>(TAB_KEYS)

/**
 * Settings page shell. Each tab is independently scrollable so a long log
 * in the install tab never pushes the header off-screen.
 */
export function SettingsPage() {
  const { params, update } = useUrlState()
  const raw = params.get("tab")
  const tab: TabKey = raw && VALID_TABS.has(raw) ? (raw as TabKey) : "overview"
  const setTab = useCallback(
    (next: TabKey) => update({ tab: next === "overview" ? null : next }),
    [update],
  )

  return (
    <div className="h-full overflow-hidden flex flex-col">
      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as TabKey)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <div className="shrink-0 border-b border-border/60 bg-background/40 px-4 py-2 md:px-6">
          <div className="overflow-x-auto">
            <TabsList variant="line" className="min-w-max">
              <TabsTrigger value="overview">概览</TabsTrigger>
              <TabsTrigger value="environment">环境</TabsTrigger>
              <TabsTrigger value="network">网络加速</TabsTrigger>
              <TabsTrigger value="models">模型下载</TabsTrigger>
              <TabsTrigger value="tagging">数据标注</TabsTrigger>
              <TabsTrigger value="ai">AI 服务商</TabsTrigger>
              <TabsTrigger value="errors">错误上报</TabsTrigger>
              <TabsTrigger value="maintenance">维护</TabsTrigger>
            </TabsList>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="overview" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "overview" && <OverviewTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="environment" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "environment" && <EnvironmentTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="network" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "network" && <NetworkTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="models" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "models" && <ModelsTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="tagging" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "tagging" && <TaggingTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="ai" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "ai" && <AIProvidersTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="errors" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "errors" && <ErrorsTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="maintenance" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-4 py-4 md:px-6 md:py-5 w-full">
                {tab === "maintenance" && <MaintenanceTab />}
              </div>
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}
