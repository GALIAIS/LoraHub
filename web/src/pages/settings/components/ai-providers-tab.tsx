import { useState } from "react"
import { Bot, Layers, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ModelsPanel } from "./ai-models-panel"
import { ProvidersPanel } from "./ai-providers-panel"
import { RoutesPanel } from "./ai-routes-panel"

export function AIProvidersTab() {
  const [activePanel, setActivePanel] = useState<
    "providers" | "models" | "routes"
  >("providers")
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant={activePanel === "providers" ? "default" : "outline"}
          onClick={() => setActivePanel("providers")}
          className="h-8"
        >
          <Settings2 className="size-3.5" /> 服务商
        </Button>
        <Button
          size="sm"
          variant={activePanel === "models" ? "default" : "outline"}
          onClick={() => setActivePanel("models")}
          className="h-8"
        >
          <Bot className="size-3.5" /> 模型
        </Button>
        <Button
          size="sm"
          variant={activePanel === "routes" ? "default" : "outline"}
          onClick={() => setActivePanel("routes")}
          className="h-8"
        >
          <Layers className="size-3.5" /> 任务路由
        </Button>
      </div>
      {activePanel === "providers" && <ProvidersPanel />}
      {activePanel === "models" && <ModelsPanel />}
      {activePanel === "routes" && <RoutesPanel />}
    </div>
  )
}
