/**
 * Environment tab — single landing page for everything install-related.
 *
 * Replaces the prior split between "依赖 / 后端管理 / 安装". Each existing
 * component is kept intact and re-rendered here as a section so the rich
 * progress UIs in InstallTab don't need a rewrite. A sticky in-page nav
 * sits on the right so the page stays navigable when all three sections
 * are unfolded.
 */
import { useEffect, useState, useRef } from "react"
import { Cpu, ServerCog, Download } from "lucide-react"
import { cn } from "@/lib/utils"
import { DependenciesTab } from "./dependencies-tab"
import { BackendsTab } from "./backends-tab"
import { InstallTab } from "./install-tab"

type SectionKey = "runtime" | "backends" | "install"

const SECTIONS: { key: SectionKey; label: string; icon: typeof Cpu; hint: string }[] = [
  {
    key: "runtime",
    label: "便携工具链",
    icon: Cpu,
    hint: "uv 与 Python 解释器",
  },
  {
    key: "backends",
    label: "训练后端",
    icon: ServerCog,
    hint: "kohya / diffusion-pipe / anima_lora 路径与 venv",
  },
  {
    key: "install",
    label: "安装与升级",
    icon: Download,
    hint: "克隆仓库、装 torch、装依赖",
  },
]

export function EnvironmentTab() {
  const [active, setActive] = useState<SectionKey>("runtime")
  const sectionRefs = useRef<Record<SectionKey, HTMLDivElement | null>>({
    runtime: null,
    backends: null,
    install: null,
  })

  // Use IntersectionObserver to highlight the in-view section in the side
  // nav. Threshold list keeps the highlight stable as the user scrolls
  // through long sub-content (the install log can be hundreds of lines).
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
        if (visible) {
          const key = (visible.target as HTMLElement).dataset.section as SectionKey
          if (key) setActive(key)
        }
      },
      { threshold: [0.15, 0.3, 0.5, 0.75], rootMargin: "-80px 0px -55% 0px" },
    )
    for (const node of Object.values(sectionRefs.current)) {
      if (node) obs.observe(node)
    }
    return () => obs.disconnect()
  }, [])

  const scrollTo = (key: SectionKey) => {
    sectionRefs.current[key]?.scrollIntoView({ behavior: "smooth", block: "start" })
    setActive(key)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_14rem] gap-6 max-w-[80rem]">
      <div className="space-y-10 min-w-0">
        <Section
          ref={(el) => {
            sectionRefs.current.runtime = el
          }}
          sectionKey="runtime"
          title="便携工具链"
          subtitle="uv + Python（python-build-standalone），由 LoraHub 管理在 .lorahub/。"
        >
          <DependenciesTab />
        </Section>

        <Section
          ref={(el) => {
            sectionRefs.current.backends = el
          }}
          sectionKey="backends"
          title="训练后端路径"
          subtitle="为 kohya / diffusion-pipe / anima_lora 配置仓库位置和 Python 解释器。"
        >
          <BackendsTab />
        </Section>

        <Section
          ref={(el) => {
            sectionRefs.current.install = el
          }}
          sectionKey="install"
          title="安装与升级"
          subtitle="克隆仓库 → 创建 venv → 装 torch → 装依赖。每一步带实时日志。"
        >
          <InstallTab />
        </Section>
      </div>

      <aside className="hidden lg:block">
        <div className="sticky top-4 space-y-1">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground/80 px-2 pb-2">
            目录
          </div>
          {SECTIONS.map((s) => {
            const Icon = s.icon
            const isActive = active === s.key
            return (
              <button
                key={s.key}
                type="button"
                onClick={() => scrollTo(s.key)}
                className={cn(
                  "w-full text-left px-3 py-2 rounded-[4px] text-xs transition-colors flex items-start gap-2",
                  isActive
                    ? "bg-primary/10 text-foreground border-l-2 border-primary"
                    : "hover:bg-muted/50 text-muted-foreground hover:text-foreground border-l-2 border-transparent",
                )}
              >
                <Icon className="size-3.5 mt-0.5 shrink-0" />
                <span className="flex-1 min-w-0">
                  <span className="block font-medium">{s.label}</span>
                  <span className="block text-[10px] text-muted-foreground/70 mt-0.5 truncate">
                    {s.hint}
                  </span>
                </span>
              </button>
            )
          })}
        </div>
      </aside>
    </div>
  )
}

interface SectionProps {
  sectionKey: SectionKey
  title: string
  subtitle: string
  children: React.ReactNode
}

const Section = (() => {
  function SectionInner(
    { sectionKey, title, subtitle, children }: SectionProps,
    ref: React.Ref<HTMLDivElement>,
  ) {
    return (
      <section
        ref={ref}
        data-section={sectionKey}
        className="scroll-mt-4"
      >
        <header className="pb-4 border-b border-border/40 mb-4">
          <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
          <p className="text-xs text-muted-foreground/85 mt-1 leading-relaxed">
            {subtitle}
          </p>
        </header>
        {children}
      </section>
    )
  }
  // forwardRef without React.forwardRef typing helper — the eslint
  // react/display-name rule only triggers when the inner function is
  // anonymous, so giving it a name here keeps the rule happy.
  return Object.assign(
    (props: SectionProps & { ref?: React.Ref<HTMLDivElement> }) => {
      const { ref, ...rest } = props
      return SectionInner(rest, ref ?? null)
    },
    { displayName: "Section" },
  )
})()
