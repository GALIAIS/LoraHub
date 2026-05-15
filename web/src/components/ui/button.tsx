import { Button as ButtonPrimitive } from "@base-ui/react/button"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center border bg-clip-padding text-sm font-medium whitespace-nowrap transition-[color,background-color,border-color,box-shadow,transform] duration-120 ease-out outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/35 active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default:
          "rounded-[2px] border-primary bg-primary text-primary-foreground shadow-[0_1px_0_rgba(255,255,255,0.12)_inset,0_10px_20px_-14px_rgba(0,0,0,0.45)] hover:bg-primary/92",
        outline:
          "rounded-[2px] border-border/80 bg-background/82 shadow-[0_1px_0_rgba(255,255,255,0.42)_inset] hover:bg-muted/80 hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/38 dark:shadow-[0_1px_0_rgba(255,255,255,0.04)_inset] dark:hover:bg-input/60",
        secondary:
          "rounded-[2px] border-border/70 bg-secondary/86 text-secondary-foreground shadow-[0_1px_0_rgba(255,255,255,0.32)_inset] hover:bg-secondary aria-expanded:bg-secondary aria-expanded:text-secondary-foreground dark:shadow-[0_1px_0_rgba(255,255,255,0.03)_inset]",
        ghost:
          "rounded-[2px] border-transparent bg-transparent hover:bg-muted/72 hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:hover:bg-muted/50",
        destructive:
          "rounded-[2px] border-destructive/18 bg-destructive/10 text-destructive hover:bg-destructive/18 focus-visible:border-destructive/40 focus-visible:ring-destructive/20 dark:bg-destructive/18 dark:hover:bg-destructive/28 dark:focus-visible:ring-destructive/40",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default:
          "h-9 gap-1.5 px-3 has-data-[icon=inline-end]:pr-2.5 has-data-[icon=inline-start]:pl-2.5",
        xs: "h-6 gap-1 px-2 text-xs has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1 px-2.5 text-[0.82rem] has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2 [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-10 gap-1.5 px-3.5 has-data-[icon=inline-end]:pr-3 has-data-[icon=inline-start]:pl-3",
        icon: "size-9",
        "icon-xs": "size-6 [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
