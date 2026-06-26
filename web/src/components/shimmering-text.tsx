import { animate, type JSAnimation } from "animejs/animation"
import { type ComponentProps, useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

export type ShimmeringTextProps = Omit<ComponentProps<"span">, "children"> & {
  text: string
  duration?: number
  isStopped?: boolean
}

export function ShimmeringText({
  text,
  duration = 1,
  isStopped = false,
  className,
  ...props
}: ShimmeringTextProps) {
  const rootRef = useRef<HTMLSpanElement | null>(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const chars = root.querySelectorAll("[data-shimmer-char]")
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    let animation: JSAnimation | null = null
    if (!isStopped && !reduce && chars.length > 0) {
      animation = animate(chars, {
        color: [
          "var(--color)",
          "var(--shimmering-color)",
          "var(--color)",
        ],
        delay: (_target: unknown, index = 0) =>
          (index * duration * 1000) / Math.max(1, chars.length),
        duration: duration * 1000,
        loop: true,
        loopDelay: chars.length * 50,
        ease: "inOutSine",
      })
    } else {
      chars.forEach((char) => {
        ;(char as HTMLElement).style.color = "var(--color)"
      })
    }
    return () => {
      animation?.revert()
    }
  }, [duration, isStopped, text])

  return (
    <span
      ref={rootRef}
      className={cn(
        "inline-flex select-none items-center leading-none",
        "[--color:var(--muted-foreground)] [--shimmering-color:var(--foreground)]",
        className,
      )}
      {...props}
    >
      {text.split("").map((char, index) => (
        <span
          aria-hidden
          className="inline-block whitespace-pre leading-none text-[var(--color)]"
          data-shimmer-char
          // biome-ignore lint/suspicious/noArrayIndexKey: static label text, order never changes
          key={index}
        >
          {char}
        </span>
      ))}
      <span className="sr-only">{text}</span>
    </span>
  )
}
