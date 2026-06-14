import * as React from "react"
import { Select as SelectPrimitive } from "@base-ui/react/select"
import { ChevronDownIcon, CheckIcon, ChevronUpIcon } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * base-ui's <Select.Value> falls back to printing the raw value when it
 * cannot find a matching label in the store-level `items` prop on
 * <Select.Root>. We don't pass that prop — every consumer just renders
 * <SelectItem> children. To bridge the gap, we build a Context-backed
 * label registry from two sources:
 *
 *   1. A static walk of the JSX children tree on every <Select> render
 *      — picks up every <SelectItem> regardless of whether the popover
 *      has ever been opened (base-ui mounts <SelectContent> lazily).
 *   2. Runtime registration from each <SelectItem>'s effect — covers
 *      cases where item children change after mount (e.g. translated
 *      label that resolves async).
 *
 * <SelectValue> looks the current value up in this map; on miss it
 * shows the placeholder rather than the raw value.
 */
interface SelectLabelRegistry {
  register: (value: string, node: React.ReactNode) => void
  unregister: (value: string) => void
  lookup: (value: string) => React.ReactNode | undefined
}

const SelectLabelContext = React.createContext<SelectLabelRegistry | null>(null)

function _normalizeKey(value: unknown): string {
  if (value == null) return ""
  if (typeof value === "string") return value
  return JSON.stringify(value)
}

/** Recursively walk a React subtree collecting `(value -> children)` pairs
 *  from every <SelectItem>. Robust to wrapping <SelectGroup> / fragments
 *  / arrays produced by {cond && <SelectItem .../>} conditionals.
 */
function _collectStaticLabels(
  node: React.ReactNode,
  out: Record<string, React.ReactNode>,
): void {
  if (node == null || typeof node === "boolean") return
  if (Array.isArray(node)) {
    for (const child of node) _collectStaticLabels(child, out)
    return
  }
  if (!React.isValidElement(node)) return
  const el = node as React.ReactElement<{
    value?: unknown
    children?: React.ReactNode
  }>
  if (el.type === SelectItem) {
    const key = _normalizeKey(el.props.value)
    if (key !== "" && key !== '""') {
      out[key] = el.props.children
    }
    return
  }
  if (el.props && "children" in el.props) {
    _collectStaticLabels(el.props.children, out)
  }
}

function Select<Value = string, Multiple extends boolean | undefined = false>({
  children,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Root<Value, Multiple>>) {
  // Pre-walk the JSX so the trigger renders the right label on first
  // paint even before base-ui has lazily mounted the popover content.
  const staticLabels = React.useMemo(() => {
    const out: Record<string, React.ReactNode> = {}
    _collectStaticLabels(children, out)
    return out
  }, [children])

  // Hold dynamic labels in a ref. Reading from a ref means the
  // <SelectValue> consumer can ask for a label without re-subscribing
  // to a state slice — and crucially, register/unregister no longer
  // call setState on every JSX child re-render. The previous version
  // dispatched one setState per <SelectItem> per render, and because
  // the registry object lived in the closure of those callbacks the
  // resulting commit would re-create the <Select> tree, which fired
  // the effects again, which hit setState, which... that's the
  // "Maximum update depth exceeded" loop datasets users were seeing.
  const dynamicLabelsRef = React.useRef<Record<string, React.ReactNode>>({})
  // Bumping this counter on each register / unregister forces a single
  // re-render so <SelectValue> can re-read the ref. We coalesce all
  // updates that happened within the same micro-task into one bump.
  const [, setLabelsRev] = React.useState(0)
  const pendingRevBumpRef = React.useRef(false)
  const scheduleRevBump = React.useCallback(() => {
    if (pendingRevBumpRef.current) return
    pendingRevBumpRef.current = true
    queueMicrotask(() => {
      pendingRevBumpRef.current = false
      setLabelsRev((r) => r + 1)
    })
  }, [])

  // Keep static labels reachable via a ref so the registry's lookup
  // doesn't have to participate in the registry's own identity.
  const staticLabelsRef = React.useRef(staticLabels)
  staticLabelsRef.current = staticLabels

  // The registry object's identity is permanent. SelectItem's register
  // effect therefore fires *exactly once* per (mount, value) pair and
  // never again on a sibling re-render — this is what closes the
  // infinite loop the previous version had.
  const registryRef = React.useRef<SelectLabelRegistry | null>(null)
  if (registryRef.current === null) {
    registryRef.current = {
      register: (value, node) => {
        const map = dynamicLabelsRef.current
        if (map[value] === node) return
        map[value] = node
        scheduleRevBump()
      },
      unregister: (value) => {
        const map = dynamicLabelsRef.current
        if (!(value in map)) return
        delete map[value]
        scheduleRevBump()
      },
      lookup: (value) => {
        const dyn = dynamicLabelsRef.current
        return value in dyn ? dyn[value] : staticLabelsRef.current[value]
      },
    }
  }

  return (
    <SelectLabelContext.Provider value={registryRef.current}>
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
      const key = _normalizeKey(value)
      // Empty value: nothing selected -> placeholder.
      if (key === "" || key === '""') {
        return placeholder ?? null
      }
      const label = registry?.lookup(key)
      if (label != null && label !== "") return label
      // Genuinely unknown value (no <SelectItem> with this value
      // anywhere in the tree) — show the placeholder rather than
      // leaking the raw token. Static label collection means a real
      // miss here implies misconfiguration upstream, not lazy mount.
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
        "flex w-fit items-center justify-between gap-1.5 rounded-[6px] border border-[var(--control-border)] bg-[var(--control-fill)] py-2 pr-2 pl-2.5 text-sm whitespace-nowrap shadow-[0_1px_0_rgba(255,255,255,0.3)_inset] transition-[color,background-color,border-color,box-shadow] duration-150 outline-none select-none hover:border-[var(--control-border-hover)] hover:bg-[var(--control-fill-hover)] focus-visible:border-ring focus-visible:bg-background focus-visible:ring-3 focus-visible:ring-ring/35 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 data-placeholder:text-muted-foreground data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-1.5 dark:shadow-[0_1px_0_rgba(255,255,255,0.04)_inset] dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
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
  // Keep the latest children in a ref so the registration effect
  // doesn't fire on every parent render. JSX children are a fresh
  // reference every render — making them part of the effect's
  // dependency array used to fire register/unregister in a loop and
  // thrash setState until React aborted with "Maximum update depth
  // exceeded".
  const childrenRef = React.useRef<React.ReactNode>(children)
  childrenRef.current = children

  React.useEffect(() => {
    if (!registry) return undefined
    const key = _normalizeKey(value)
    if (key === "" || key === '""') return undefined
    registry.register(key, childrenRef.current)
    return () => registry.unregister(key)
  }, [registry, value])

  return (
    <SelectPrimitive.Item
      data-slot="select-item"
      value={value}
      className={cn(
        "relative flex w-full cursor-default items-center gap-1.5 rounded-[6px] py-1.5 pr-8 pl-1.5 text-sm outline-hidden select-none focus:bg-[var(--state-hover)] focus:text-foreground data-disabled:pointer-events-none data-disabled:opacity-50 data-highlighted:bg-[var(--state-hover)] [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 *:[span]:last:flex *:[span]:last:items-center *:[span]:last:gap-2",
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
