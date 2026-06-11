import type * as React from "react"
import { cn } from "@/lib/utils"

interface WorkbenchSplitLayoutProps {
  sidebarOpen?: boolean
  sidebarWidth?: string
  sidebar: React.ReactNode
  children: React.ReactNode
  className?: string
  asideClassName?: string
  mainClassName?: string
  transitionClassName?: string
}

export function WorkbenchSplitLayout({
  sidebarOpen = true,
  sidebarWidth = "minmax(260px,300px)",
  sidebar,
  children,
  className,
  asideClassName,
  mainClassName,
  transitionClassName = "duration-200",
}: WorkbenchSplitLayoutProps) {
  return (
    <div
      className={cn(
        "grid h-full min-h-0 grid-rows-[1fr] overflow-hidden transition-[grid-template-columns] ease-out",
        transitionClassName,
        className,
      )}
      style={{
        gridTemplateColumns: sidebarOpen
          ? `${sidebarWidth} minmax(0,1fr)`
          : "0px minmax(0,1fr)",
      }}
    >
      <aside
        className={cn(
          "shiro-page-aside flex min-h-0 min-w-0 flex-col overflow-hidden transition-opacity duration-200",
          !sidebarOpen && "pointer-events-none opacity-0",
          asideClassName,
        )}
        aria-hidden={!sidebarOpen}
      >
        {sidebar}
      </aside>

      <section
        className={cn(
          "relative flex min-h-0 min-w-0 flex-col overflow-hidden bg-background/60",
          mainClassName,
        )}
      >
        {children}
      </section>
    </div>
  )
}
