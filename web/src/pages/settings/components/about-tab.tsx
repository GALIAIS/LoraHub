import { AlertTriangle, ExternalLink, Github, Heart, Scale } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useVersionInfo } from "@/hooks/use-version-info"

const REPO_URL = "https://github.com/GALIAIS/LoraHub"
const ISSUES_URL = `${REPO_URL}/issues`
const RELEASES_URL = `${REPO_URL}/releases`
const LICENSE_URL = `${REPO_URL}/blob/main/LICENSE`
const QQ_GROUP_URL = "https://qm.qq.com/q/YfjMJqatKQ"

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
 * About tab — project metadata, links, and the *pair* of versions
 * (frontend bundle vs backend service) so a stale `web/dist` after a
 * `git pull` is immediately visible. The previous single-badge layout
 * pulled only the backend version from `/api/health`, which masked the
 * exact mismatch the install/update flow can leave behind when it
 * skips a SPA rebuild.
 */
export function AboutTab() {
  const {
    frontendDisplay,
    backendDisplay,
    frontend,
    backend,
    mismatch,
    loading,
  } = useVersionInfo()

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
            <div className="flex items-center gap-1.5">
              <Badge
                variant="outline"
                className={
                  mismatch
                    ? "font-mono text-[11px] border-amber-500/60 text-amber-700 dark:text-amber-400"
                    : "font-mono text-[11px]"
                }
                title={`前端构建版本 (git describe，编译期注入)\n原始: ${frontend}`}
              >
                Frontend {frontendDisplay}
              </Badge>
              <Badge
                variant="outline"
                className={
                  mismatch
                    ? "font-mono text-[11px] border-amber-500/60 text-amber-700 dark:text-amber-400"
                    : "font-mono text-[11px]"
                }
                title={`后端运行版本 (lorahub.__version__，hatch-vcs 解析 git tag)\n原始: ${backend ?? (loading ? "loading…" : "unknown")}`}
              >
                Backend {loading && backendDisplay === "?" ? "…" : backendDisplay}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed">
          {mismatch && (
            <div
              className="flex items-start gap-2 rounded-[4px] border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] leading-relaxed"
            >
              <AlertTriangle className="size-4 mt-0.5 shrink-0 text-amber-700 dark:text-amber-400" />
              <div className="space-y-1">
                <div className="font-medium text-amber-800 dark:text-amber-300">
                  前后端 commit 不一致
                </div>
                <div className="text-amber-900/80 dark:text-amber-200/85">
                  前端 bundle 内嵌的 git sha 与后端运行时的 sha 不同。
                  后端代码更新后，需要重建 <code className="font-mono">web/dist/</code>{" "}
                  以匹配当前 API。
                </div>
                <div className="text-amber-900/80 dark:text-amber-200/85 pt-1">
                  重建命令：
                </div>
                <ul className="list-disc pl-5 space-y-0.5 font-mono text-[11px] text-amber-900/85 dark:text-amber-200/90">
                  <li>
                    <code>lorahub manage build</code>
                    {" "}: 仅重建前端
                  </li>
                  <li>
                    <code>lorahub manage update</code>
                    {" "}: 拉新代码 + 重装依赖 + 重建前端
                  </li>
                  <li>
                    <code>scripts\run.bat dev</code>
                    {" "}: Vite 热更新（开发模式）
                  </li>
                </ul>
              </div>
            </div>
          )}
          <p>
            LoraHub 把 kohya-ss/sd-scripts 与 tdrussell/diffusion-pipe
            两套训练后端、配置编辑器、数据集预处理、自动标注与作业调度统一在
            单进程 FastAPI 服务后，前端通过本地 HTTP 调用，运行数据保存在本机。
          </p>
          <p className="text-muted-foreground">
            开源 AGPL-3.0 协议; 训练数据、配置、运行结果全部存放在用户的本地工作目录,
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
                QQ群
              </dt>
              <dd>
                <ExternalAnchor href={QQ_GROUP_URL}>1098319682</ExternalAnchor>
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
            可独立替换的模块化设计，后端与前端解耦
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
