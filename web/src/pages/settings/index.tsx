import { useState } from "react"
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

type TabKey =
  | "overview"
  | "environment"
  | "network"
  | "models"
  | "tagging"
  | "ai"
  | "errors"
  | "maintenance"

/**
 * Settings page shell. Each tab is independently scrollable so a long log
 * in the install tab never pushes the header off-screen.
 */
export function SettingsPage() {
  const [tab, setTab] = useState<TabKey>("overview")

  return (
    <div className="h-full overflow-hidden flex flex-col">
      <header className="shrink-0 px-8 pt-7 pb-4 space-y-1 border-b border-border/60 bg-background/40">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          工作台
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">设置</h1>
        <p className="text-sm text-muted-foreground">
          工作区级别的默认值。配置文件中的同名字段会按任务覆盖;环境变量
          (LORAHUB_*) 优先级最高。
        </p>
      </header>

      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as TabKey)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <div className="px-8 pt-3 pb-2 border-b border-border/60 bg-background/40 shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="environment">环境</TabsTrigger>
            <TabsTrigger value="network">网络加速</TabsTrigger>
            <TabsTrigger value="models">模型下载</TabsTrigger>
            <TabsTrigger value="tagging">标注</TabsTrigger>
            <TabsTrigger value="ai">AI 服务商</TabsTrigger>
            <TabsTrigger value="errors">错误上报</TabsTrigger>
            <TabsTrigger value="maintenance">维护</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="overview" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "overview" && <OverviewTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="environment" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "environment" && <EnvironmentTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="network" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "network" && <NetworkTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="models" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "models" && <ModelsTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="tagging" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "tagging" && <TaggingTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="ai" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "ai" && <AIProvidersTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="errors" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "errors" && <ErrorsTab />}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="maintenance" className="h-full">
            <div className="h-full overflow-y-auto">
              <div className="px-8 py-6 w-full">
                {tab === "maintenance" && <MaintenanceTab />}
              </div>
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}
