import { useQuery } from "@tanstack/react-query"
import { ExternalLink, Github, Heart, Scale } from "lucide-react"
import { api } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

const REPO_URL = "https://github.com/GALIAIS/LoraHub"
const ISSUES_URL = `${REPO_URL}/issues`
const RELEASES_URL = `${REPO_URL}/releases`
const LICENSE_URL = `${REPO_URL}/blob/main/LICENSE`

interface ExternalAnchorProps {
  href: string
  children: React.ReactNode
}

function ExternalAnchor({ href, children }: ExternalAnchorProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-foreground hover:underline underline-offset-2"
    >
      {children}
      <ExternalLink className="size-3 opacity-60" />
    </a>
  )
}

/**
 * About tab — project metadata, links, and version pulled live from
 * `/api/health` so the user always sees the version they're actually
 * connected to (not whatever the SPA was built with).
 */
export function AboutTab() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: 60_000,
  })

  const version = healthQuery.data?.version ?? "—"

  return (
    <div className="space-y-5 max-w-3xl">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="space-y-1">
              <CardTitle className="text-xl tracking-tight">LoraHub</CardTitle>
              <CardDescription>
                面向扩散模型的 LoRA 训练工作台
              </CardDescription>
            </div>
            <Badge variant="outline" className="font-mono text-[11px]">
              v{version}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed">
          <p>
            LoraHub 把 kohya-ss/sd-scripts 与 tdrussell/diffusion-pipe
            两套训练后端、配方编辑器、数据集预处理、自动标注与作业调度统一在
            单进程 FastAPI 服务后,前端通过纯本地 HTTP 调用,无需登录、无云端依赖。
          </p>
          <p className="text-muted-foreground">
            开源 AGPL-3.0 协议; 训练数据、配方、运行结果全部存放在用户的本地工作目录,
            可随时 git 跟踪、备份或迁移。
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">链接</CardTitle>
          <CardDescription>仓库、问题反馈与发布说明</CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 sm:grid-cols-2 text-sm">
            <div className="space-y-1">
              <dt className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                <Github className="size-3" />
                仓库
              </dt>
              <dd>
                <ExternalAnchor href={REPO_URL}>
                  github.com/GALIAIS/LoraHub
                </ExternalAnchor>
              </dd>
            </div>
            <div className="space-y-1">
              <dt className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                <Heart className="size-3" />
                问题反馈
              </dt>
              <dd>
                <ExternalAnchor href={ISSUES_URL}>Issues</ExternalAnchor>
              </dd>
            </div>
            <div className="space-y-1">
              <dt className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                发布
              </dt>
              <dd>
                <ExternalAnchor href={RELEASES_URL}>Releases</ExternalAnchor>
              </dd>
            </div>
            <div className="space-y-1">
              <dt className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                <Scale className="size-3" />
                协议
              </dt>
              <dd>
                <ExternalAnchor href={LICENSE_URL}>AGPL-3.0</ExternalAnchor>
              </dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">技术栈</CardTitle>
          <CardDescription>
            可独立替换的模块化设计,后端与前端解耦
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm">
          <ul className="grid gap-1.5 sm:grid-cols-2 text-[13px]">
            <li>
              <span className="text-muted-foreground">服务端:</span>
              <span className="ml-2 font-mono">FastAPI · Pydantic · SQLite</span>
            </li>
            <li>
              <span className="text-muted-foreground">前端:</span>
              <span className="ml-2 font-mono">React · Vite · TanStack Query</span>
            </li>
            <li>
              <span className="text-muted-foreground">训练后端:</span>
              <span className="ml-2 font-mono">sd-scripts · diffusion-pipe</span>
            </li>
            <li>
              <span className="text-muted-foreground">运行时:</span>
              <span className="ml-2 font-mono">Python 3.11+ · uv venv</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
