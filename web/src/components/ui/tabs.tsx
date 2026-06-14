import { Tabs as TabsPrimitive } from "@base-ui/react/tabs"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

function Tabs({
  className,
  orientation = "horizontal",
  ...props
}: TabsPrimitive.Root.Props) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      data-orientation={orientation}
      className={cn(
        "group/tabs flex gap-2 data-horizontal:flex-col",
        className
      )}
      {...props}
    />
  )
}

const tabsListVariants = cva(
  "group/tabs-list inline-flex w-fit items-center justify-center rounded-[8px] border border-[var(--control-border)] bg-[var(--control-fill)] p-[3px] text-muted-foreground shadow-[0_1px_0_rgba(255,255,255,0.28)_inset] group-data-horizontal/tabs:min-h-9 group-data-vertical/tabs:h-fit group-data-vertical/tabs:flex-col data-[variant=line]:border-0 data-[variant=line]:bg-transparent data-[variant=line]:p-0 data-[variant=line]:shadow-none",
  {
    variants: {
      variant: {
        default: "",
        line: "gap-1 bg-transparent",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function TabsList({
  className,
  variant = "default",
  ...props
}: TabsPrimitive.List.Props & VariantProps<typeof tabsListVariants>) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      data-variant={variant}
      className={cn(tabsListVariants({ variant }), className)}
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-trigger"
      className={cn(
        "relative inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-[6px] border border-transparent px-2.5 py-1 text-sm font-medium whitespace-nowrap text-foreground/62 transition-[color,background-color,border-color,box-shadow] duration-150 group-data-vertical/tabs:w-full group-data-vertical/tabs:justify-start hover:bg-[var(--state-hover)] hover:text-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/35 focus-visible:outline-1 focus-visible:outline-ring disabled:pointer-events-none disabled:opacity-50 has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 aria-disabled:pointer-events-none aria-disabled:opacity-50 dark:text-muted-foreground dark:hover:text-foreground group-data-[variant=default]/tabs-list:data-active:shadow-[0_1px_0_rgba(255,255,255,0.42)_inset] group-data-[variant=line]/tabs-list:data-active:shadow-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        "group-data-[variant=line]/tabs-list:bg-transparent group-data-[variant=line]/tabs-list:data-active:bg-transparent dark:group-data-[variant=line]/tabs-list:data-active:border-transparent dark:group-data-[variant=line]/tabs-list:data-active:bg-transparent",
        "data-active:border-[var(--control-border-hover)] data-active:bg-background/90 data-active:text-foreground dark:data-active:bg-input/48 dark:data-active:text-foreground",
        "before:absolute before:inset-0 before:border before:border-transparent before:opacity-0 before:transition-opacity before:content-[''] group-data-[variant=line]/tabs-list:data-active:before:border-foreground/24 group-data-[variant=line]/tabs-list:data-active:before:opacity-100",
        className
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-content"
      className={cn("flex-1 text-sm outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent, tabsListVariants }
