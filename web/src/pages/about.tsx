import { AboutTab } from "./settings/components/about-tab"

/**
 * Standalone About page — same content as the AboutTab card, but mounted
 * as its own top-level route under the main sidebar (below 设置).
 */
export function AboutPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="px-4 py-4 md:px-6 md:py-5 w-full">
        <AboutTab />
      </div>
    </div>
  )
}
