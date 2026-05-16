import { useState } from "react"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { OverviewTab } from "./components/overview-tab"
import { BackendsTab } from "./components/backends-tab"
import { InstallTab } from "./components/install-tab"
import { TaggingTab } from "./components/tagging-tab"
import { NetworkTab } from "./components/network-tab"
import { ModelsTab } from "./components/models-tab"
import { DependenciesTab } from "./components/dependencies-tab"
import { AboutTab } from "./components/about-tab"

type TabKey =
  | "overview"
  | "dependencies"
  | "backends"
  | "install"
  | "network"
  | "models"
  | "tagging"

/**
 * Settings page shell. The active tab's content scrolls together with a
 * page-wide About footer below it, so users always have the project
 * intro and links a scroll away regardless of which tab they're on.
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
            <TabsTrigger value="dependencies">依赖</TabsTrigger>
            <TabsTrigger value="backends">后端管理</TabsTrigger>
            <TabsTrigger value="install">安装</TabsTrigger>
            <TabsTrigger value="network">网络加速</TabsTrigger>
            <TabsTrigger value="models">模型下载</TabsTrigger>
            <TabsTrigger value="tagging">标注</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="px-8 py-6 w-full space-y-8">
            <TabsContent value="overview"><OverviewTab /></TabsContent>
            <TabsContent value="dependencies"><DependenciesTab /></TabsContent>
            <TabsContent value="backends"><BackendsTab /></TabsContent>
            <TabsContent value="install"><InstallTab /></TabsContent>
            <TabsContent value="network"><NetworkTab /></TabsContent>
            <TabsContent value="models"><ModelsTab /></TabsContent>
            <TabsContent value="tagging"><TaggingTab /></TabsContent>

            <div className="pt-4 border-t border-border/50">
              <AboutTab />
            </div>
          </div>
        </div>
      </Tabs>
    </div>
  )
}
