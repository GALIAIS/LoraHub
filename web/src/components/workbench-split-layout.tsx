import * as React from "react"
import { PanelLeftOpen } from "lucide-react"
import { cn } from "@/lib/utils"
import { useIsMobile } from "@/hooks/use-mobile"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"

interface WorkbenchSplitLayoutProps {
  sidebarOpen?: boolean
  sidebarWidth?: string
  sidebar: React.ReactNode
  children: React.ReactNode
  className?: string
  asideClassName?: string
  mainClassName?: string
  transitionClassName?: string
  mobileSidebarTitle?: string
  mobileSidebarDescription?: string
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
  mobileSidebarTitle = "工作区侧栏",
  mobileSidebarDescription = "筛选、选择和管理当前页面内容。",
}: WorkbenchSplitLayoutProps) {
  const isMobile = useIsMobile()
  const [mobileOpen, setMobileOpen] = React.useState(false)

  return (
    <>
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent
          side="left"
          className="w-[min(92vw,22rem)] gap-0 overflow-hidden rounded-none p-0 md:hidden"
        >
          <SheetHeader className="sr-only">
            <SheetTitle>{mobileSidebarTitle}</SheetTitle>
            <SheetDescription>{mobileSidebarDescription}</SheetDescription>
          </SheetHeader>
          <div className="flex h-full min-h-0 flex-col">{sidebar}</div>
        </SheetContent>
      </Sheet>

      <div
        className={cn(
          "grid h-full min-h-0 grid-rows-[1fr] overflow-hidden transition-[grid-template-columns] ease-out",
          transitionClassName,
          className,
        )}
        style={{
          gridTemplateColumns: isMobile
            ? "minmax(0,1fr)"
            : sidebarOpen
              ? `${sidebarWidth} minmax(0,1fr)`
              : "0px minmax(0,1fr)",
        }}
      >
        <aside
          className={cn(
            "shiro-page-aside hidden min-h-0 min-w-0 flex-col overflow-hidden transition-opacity duration-200 md:flex",
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
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setMobileOpen(true)}
            className="absolute left-3 top-3 z-20 h-7 gap-1 bg-background/92 px-2 text-[11px] shadow-none backdrop-blur md:hidden"
            title={mobileSidebarTitle}
          >
            <PanelLeftOpen className="size-3.5" />
            <span>侧栏</span>
          </Button>
          {children}
        </section>
      </div>
    </>
  )
}
