/**
 * Compact Markdown renderer tuned for AI analysis cards inside LoraHub.
 *
 * `react-markdown` + GFM gives us tables, task lists, strikethrough and
 * autolinks. We don't pull in `@tailwindcss/typography` because the
 * standard `prose` scale is way too generous for our 12px panels — the
 * inline component overrides below keep paragraph rhythm tight while
 * still rendering structured prose the user expects from a model
 * response.
 */
import { memo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"

interface MarkdownProps {
  source: string
  className?: string
}

export const Markdown = memo(function Markdown({
  source,
  className,
}: MarkdownProps) {
  return (
    <div
      className={cn(
        "text-[12.5px] leading-[1.65] text-foreground/90",
        "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node: _node, ...p }) => (
            <h1
              className="mt-4 mb-2 text-[15px] font-semibold tracking-tight text-foreground"
              {...p}
            />
          ),
          h2: ({ node: _node, ...p }) => (
            <h2
              className="mt-4 mb-2 text-[14px] font-semibold tracking-tight text-foreground border-b border-border/60 pb-1"
              {...p}
            />
          ),
          h3: ({ node: _node, ...p }) => (
            <h3
              className="mt-3 mb-1.5 text-[13px] font-semibold text-foreground"
              {...p}
            />
          ),
          h4: ({ node: _node, ...p }) => (
            <h4
              className="mt-2.5 mb-1 text-[12.5px] font-semibold text-foreground/90"
              {...p}
            />
          ),
          p: ({ node: _node, ...p }) => (
            <p className="my-2 text-foreground/90" {...p} />
          ),
          ul: ({ node: _node, ...p }) => (
            <ul className="my-2 ml-5 list-disc space-y-1 marker:text-muted-foreground/60" {...p} />
          ),
          ol: ({ node: _node, ...p }) => (
            <ol className="my-2 ml-5 list-decimal space-y-1 marker:text-muted-foreground/60" {...p} />
          ),
          li: ({ node: _node, ...p }) => (
            <li className="text-foreground/90" {...p} />
          ),
          strong: ({ node: _node, ...p }) => (
            <strong className="font-semibold text-foreground" {...p} />
          ),
          em: ({ node: _node, ...p }) => (
            <em className="italic text-foreground/90" {...p} />
          ),
          a: ({ node: _node, href, ...p }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline-offset-2 hover:underline"
              {...p}
            />
          ),
          blockquote: ({ node: _node, ...p }) => (
            <blockquote
              className="my-3 border-l-2 border-border/70 bg-muted/30 px-3 py-1.5 text-foreground/85 italic"
              {...p}
            />
          ),
          hr: () => <hr className="my-4 border-border/60" />,
          code: ({ node: _node, className, children, ...p }) => {
            const isBlock = /language-/.test(className ?? "")
            if (isBlock) {
              return (
                <code
                  className={cn(
                    "block whitespace-pre-wrap font-mono text-[11.5px] leading-[1.55] text-foreground/90",
                    className,
                  )}
                  {...p}
                >
                  {children}
                </code>
              )
            }
            return (
              <code
                className="rounded-[3px] border border-border/50 bg-muted/50 px-1 py-px font-mono text-[0.92em] text-foreground/95"
                {...p}
              >
                {children}
              </code>
            )
          },
          pre: ({ node: _node, ...p }) => (
            <pre
              className="my-3 overflow-x-auto rounded-[5px] border border-border/60 bg-muted/40 p-3"
              {...p}
            />
          ),
          table: ({ node: _node, ...p }) => (
            <div className="my-3 overflow-x-auto rounded-[5px] border border-border/60">
              <table
                className="w-full border-collapse text-[11.5px] tabular-nums"
                {...p}
              />
            </div>
          ),
          thead: ({ node: _node, ...p }) => (
            <thead className="bg-muted/40" {...p} />
          ),
          th: ({ node: _node, ...p }) => (
            <th
              className="border-b border-border/60 px-2.5 py-1.5 text-left font-semibold text-foreground/90"
              {...p}
            />
          ),
          td: ({ node: _node, ...p }) => (
            <td
              className="border-b border-border/30 px-2.5 py-1.5 align-top text-foreground/85"
              {...p}
            />
          ),
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  )
})
