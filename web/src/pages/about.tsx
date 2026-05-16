import { AboutTab } from "./settings/components/about-tab"

/**
 * Standalone About page — same content as the AboutTab card, but mounted
 * as its own top-level route under the main sidebar (below 设置).
 */
export function AboutPage() {
  return (
    <div className="h-full overflow-y-auto">
      <header className="px-8 pt-7 pb-4 space-y-1 border-b border-border/60 bg-background/40">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          工作台
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">关于</h1>
        <p className="text-sm text-muted-foreground">
          项目介绍、版本与仓库链接。
        </p>
      </header>
      <div className="px-8 py-6 w-full">
        <AboutTab />
      </div>
    </div>
  )
}
