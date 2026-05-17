import * as React from "react"
import { Select as SelectPrimitive } from "@base-ui/react/select"
import { ChevronDownIcon, CheckIcon, ChevronUpIcon } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * base-ui's <Select.Value> falls back to printing the raw value when it
 * cannot find a matching label in the store-level `items` prop on
 * <Select.Root>. We don't pass that prop — every consumer just renders
 * <SelectItem> children. To bridge the gap, we maintain a small
 * Context-backed registry: each <SelectItem> publishes its
 * `value -> rendered children` pair on mount, and our <SelectValue>
 * looks the current value up in that map. If the lookup misses (item
 * not yet mounted, or genuinely unknown value), we fall back to the
 * placeholder rather than the raw value.
 */
interface SelectLabelRegistry {
  register: (value: string, node: React.ReactNode) => void
  unregister: (value: string) => void
  lookup: (value: string) => React.ReactNode | undefined
}

const SelectLabelContext = React.createContext<SelectLabelRegistry | null>(null)

function Select<Value = string, Multiple extends boolean | undefined = false>({
  children,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Root<Value, Multiple>>) {
  const [labels, setLabels] = React.useState<Record<string, React.ReactNode>>({})
  const registry = React.useMemo<SelectLabelRegistry>(
    () => ({
      register: (value, node) =>
        setLabels((prev) =>
          prev[value] === node ? prev : { ...prev, [value]: node },
        ),
      unregister: (value) =>
        setLabels((prev) => {
          if (!(value in prev)) return prev
          const next = { ...prev }
          delete next[value]
          return next
        }),
      lookup: (value) => labels[value],
    }),
    [labels],
  )
  return (
    <SelectLabelContext.Provider value={registry}>
      <SelectPrimitive.Root {...props}>{children}</SelectPrimitive.Root>
    </SelectLabelContext.Provider>
  )
}

function SelectGroup({ className, ...props }: SelectPrimitive.Group.Props) {
  return (
    <SelectPrimitive.Group
      data-slot="select-group"
      className={cn("scroll-my-1 p-1", className)}
      {...props}
    />
  )
}

function SelectValue({
  className,
  children: childrenProp,
  placeholder,
  ...props
}: SelectPrimitive.Value.Props & { placeholder?: React.ReactNode }) {
  const registry = React.useContext(SelectLabelContext)
  const renderChildren = React.useCallback(
    (value: unknown) => {
      // Caller-supplied children take precedence (escape hatch for
      // multi-select / custom rendering).
      if (typeof childrenProp === "function") {
        return childrenProp(value as never)
      }
      if (childrenProp != null && childrenProp !== "") {
        return childrenProp
      }
      const key =
        value == null
          ? ""
          : typeof value === "string"
            ? value
            : JSON.stringify(value)
      // Empty value: nothing selected -> placeholder.
      if (key === "" || key === '""') {
        return placeholder ?? null
      }
      const label = registry?.lookup(key)
      if (label != null && label !== "") return label
      // Last resort: don't print the raw value; show placeholder.
      return placeholder ?? null
    },
    [childrenProp, placeholder, registry],
  )
  return (
    <SelectPrimitive.Value
      data-slot="select-value"
      className={cn("flex flex-1 text-left", className)}
      placeholder={placeholder as string | undefined}
      {...props}
    >
      {renderChildren}
    </SelectPrimitive.Value>
  )
}

function SelectTrigger({
  className,
  size = "default",
  children,
  ...props
}: SelectPrimitive.Trigger.Props & {
  size?: "sm" | "default"
}) {
  return (
    <SelectPrimitive.Trigger
      data-slot="select-trigger"
      data-size={size}
      className={cn(
        "flex w-fit items-center justify-between gap-1.5 rounded-[2px] border border-input bg-background/76 py-2 pr-2 pl-2.5 text-sm whitespace-nowrap shadow-[0_1px_0_rgba(255,255,255,0.3)_inset] transition-colors outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/35 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 data-placeholder:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-1.5 dark:bg-input/30 dark:shadow-[0_1px_0_rgba(255,255,255,0.04)_inset] dark:hover:bg-input/50 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon
        render={
          <ChevronDownIcon className="pointer-events-none size-4 text-muted-foreground" />
        }
      />
    </SelectPrimitive.Trigger>
  )
}

function SelectContent({
  className,
  children,
  side = "bottom",
  sideOffset = 4,
  align = "center",
  alignOffset = 0,
  alignItemWithTrigger = true,
  ...props
}: SelectPrimitive.Popup.Props &
  Pick<
    SelectPrimitive.Positioner.Props,
    "align" | "alignOffset" | "side" | "sideOffset" | "alignItemWithTrigger"
  >) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Positioner
        side={side}
        sideOffset={sideOffset}
        align={align}
        alignOffset={alignOffset}
        alignItemWithTrigger={alignItemWithTrigger}
        className="isolate z-50"
      >
        <SelectPrimitive.Popup
          data-slot="select-content"
          data-align-trigger={alignItemWithTrigger}
          className={cn(
            "shiro-surface relative isolate z-50 max-h-(--available-height) w-(--anchor-width) min-w-36 origin-(--transform-origin) overflow-x-hidden overflow-y-auto bg-popover text-popover-foreground will-change-[transform,opacity] duration-100 data-[align-trigger=true]:animate-none data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
            className,
          )}
          {...props}
        >
          <SelectScrollUpButton />
          <SelectPrimitive.List>{children}</SelectPrimitive.List>
          <SelectScrollDownButton />
        </SelectPrimitive.Popup>
      </SelectPrimitive.Positioner>
    </SelectPrimitive.Portal>
  )
}

function SelectLabel({ className, ...props }: SelectPrimitive.GroupLabel.Props) {
  return (
    <SelectPrimitive.GroupLabel
      data-slot="select-label"
      className={cn("px-1.5 py-1 text-xs text-muted-foreground", className)}
      {...props}
    />
  )
}

function SelectItem({
  className,
  children,
  value,
  ...props
}: SelectPrimitive.Item.Props) {
  const registry = React.useContext(SelectLabelContext)
  React.useEffect(() => {
    if (!registry) return undefined
    const key =
      value == null
        ? ""
        : typeof value === "string"
          ? value
          : JSON.stringify(value)
    if (key === "" || key === '""') return undefined
    registry.register(key, children)
    return () => registry.unregister(key)
  }, [registry, value, children])
  return (
    <SelectPrimitive.Item
      data-slot="select-item"
      value={value}
      className={cn(
        "relative flex w-full cursor-default items-center gap-1.5 rounded-[2px] py-1.5 pr-8 pl-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 *:[span]:last:flex *:[span]:last:items-center *:[span]:last:gap-2",
        className,
      )}
      {...props}
    >
      <SelectPrimitive.ItemText className="flex flex-1 shrink-0 gap-2 whitespace-nowrap">
        {children}
      </SelectPrimitive.ItemText>
      <SelectPrimitive.ItemIndicator
        render={
          <span className="pointer-events-none absolute right-2 flex size-4 items-center justify-center" />
        }
      >
        <CheckIcon className="pointer-events-none" />
      </SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
  )
}

function SelectSeparator({ className, ...props }: SelectPrimitive.Separator.Props) {
  return (
    <SelectPrimitive.Separator
      data-slot="select-separator"
      className={cn("pointer-events-none -mx-1 my-1 h-px bg-border", className)}
      {...props}
    />
  )
}

function SelectScrollUpButton({
  className,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollUpArrow>) {
  return (
    <SelectPrimitive.ScrollUpArrow
      data-slot="select-scroll-up-button"
      className={cn(
        "top-0 z-10 flex w-full cursor-default items-center justify-center bg-popover py-1 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    >
      <ChevronUpIcon />
    </SelectPrimitive.ScrollUpArrow>
  )
}

function SelectScrollDownButton({
  className,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.ScrollDownArrow>) {
  return (
    <SelectPrimitive.ScrollDownArrow
      data-slot="select-scroll-down-button"
      className={cn(
        "bottom-0 z-10 flex w-full cursor-default items-center justify-center bg-popover py-1 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    >
      <ChevronDownIcon />
    </SelectPrimitive.ScrollDownArrow>
  )
}

export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectScrollDownButton,
  SelectScrollUpButton,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
}
