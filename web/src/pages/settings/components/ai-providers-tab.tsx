import { useState } from "react"
import { Bot, Layers, Settings2 } from "lucide-react"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { ModelsPanel } from "./ai-models-panel"
import { ProvidersPanel } from "./ai-providers-panel"
import { RoutesPanel } from "./ai-routes-panel"

type AIPanel = "providers" | "models" | "routes"

export function AIProvidersTab() {
  const [activePanel, setActivePanel] = useState<AIPanel>("providers")
  return (
    <Tabs
      value={activePanel}
      onValueChange={(v) => setActivePanel(v as AIPanel)}
      className="space-y-4"
    >
      <div className="overflow-x-auto">
        <TabsList className="min-w-max">
          <TabsTrigger value="providers">
            <Settings2 className="size-3.5" /> 服务商
          </TabsTrigger>
          <TabsTrigger value="models">
            <Bot className="size-3.5" /> 模型
          </TabsTrigger>
          <TabsTrigger value="routes">
            <Layers className="size-3.5" /> 任务路由
          </TabsTrigger>
        </TabsList>
      </div>
      <TabsContent value="providers">
        {activePanel === "providers" && <ProvidersPanel />}
      </TabsContent>
      <TabsContent value="models">
        {activePanel === "models" && <ModelsPanel />}
      </TabsContent>
      <TabsContent value="routes">
        {activePanel === "routes" && <RoutesPanel />}
      </TabsContent>
    </Tabs>
  )
}
