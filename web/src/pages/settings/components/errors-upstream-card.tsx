import { useEffect, useRef, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Network } from "lucide-react"
import { toast } from "sonner"

import { api, errorReportsApi, type SettingsState } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type UpstreamChannel = SettingsState["error_upstream_channel"]

export function UpstreamStatusBadge({ status }: { status: string }) {
  const tone =
    status === "sent"
      ? "text-emerald-600 dark:text-emerald-400"
      : status === "failed"
        ? "text-destructive"
        : "text-cyan-700 dark:text-cyan-400"
  const labelMap: Record<string, string> = {
    queued: "排队中",
    retrying: "重试中",
    sent: "已发送",
    failed: "发送失败",
    skipped: "已跳过",
  }
  return (
    <Badge variant="outline" className={`rounded-[2px] ${tone}`}>
      {labelMap[status] ?? status}
    </Badge>
  )
}

export function UpstreamConfigCard() {
  const qc = useQueryClient()
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: () => api.getSettings(),
    staleTime: 30_000,
  })
  const cfg = settings.data?.settings

  const [draft, setDraft] = useState({
    error_upstream_channel: "off" as UpstreamChannel,
    error_upstream_gitlab_base_url: "",
    error_upstream_gitlab_repo: "",
    error_upstream_gitlab_token: "",
    error_upstream_webhook_url: "",
    error_upstream_webhook_auth_header: "",
    error_upstream_auto_severity:
      "error" as SettingsState["error_upstream_auto_severity"],
  })
  const [saving, setSaving] = useState(false)
  const [probing, setProbing] = useState(false)
  const hydratedRef = useRef<string | null>(null)

  useEffect(() => {
    if (!cfg) return
    const stamp = JSON.stringify({
      ch: cfg.error_upstream_channel ?? "off",
      base: cfg.error_upstream_gitlab_base_url ?? "",
      repo: cfg.error_upstream_gitlab_repo ?? "",
      tok: cfg.error_upstream_gitlab_token ?? "",
      hook: cfg.error_upstream_webhook_url ?? "",
      auth: cfg.error_upstream_webhook_auth_header ?? "",
      sev: cfg.error_upstream_auto_severity ?? "error",
    })
    if (hydratedRef.current === stamp) return
    hydratedRef.current = stamp
    setDraft({
      error_upstream_channel: (cfg.error_upstream_channel ??
        "off") as UpstreamChannel,
      error_upstream_gitlab_base_url:
        cfg.error_upstream_gitlab_base_url ?? "",
      error_upstream_gitlab_repo: cfg.error_upstream_gitlab_repo ?? "",
      error_upstream_gitlab_token: cfg.error_upstream_gitlab_token ?? "",
      error_upstream_webhook_url: cfg.error_upstream_webhook_url ?? "",
      error_upstream_webhook_auth_header:
        cfg.error_upstream_webhook_auth_header ?? "",
      error_upstream_auto_severity:
        (cfg.error_upstream_auto_severity ??
          "error") as SettingsState["error_upstream_auto_severity"],
    })
  }, [cfg])

  const onSave = async () => {
    setSaving(true)
    try {
      const fresh = await api.updateSettings(draft as Partial<SettingsState>)
      qc.setQueryData(["settings"], fresh)
      toast.success("已保存远端上报配置")
    } catch (e) {
      toast.error("保存失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setSaving(false)
    }
  }

  const onProbe = async () => {
    setProbing(true)
    try {
      const res = await errorReportsApi.upstreamHealth({
        channel: draft.error_upstream_channel,
        gitlab_base_url: draft.error_upstream_gitlab_base_url,
        gitlab_repo: draft.error_upstream_gitlab_repo,
        gitlab_token: draft.error_upstream_gitlab_token,
        webhook_url: draft.error_upstream_webhook_url,
        webhook_auth_header: draft.error_upstream_webhook_auth_header,
      })
      if (res.ok) {
        toast.success("远端连通正常", {
          description: res.url ?? `channel=${res.channel}`,
        })
      } else {
        toast.error("远端连通失败", {
          description: res.error ?? "(未提供详情)",
          duration: 14_000,
        })
      }
    } catch (e) {
      toast.error("连通测试出错", {
        description: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setProbing(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Network className="size-4" />
          远端上报通道
        </CardTitle>
        <CardDescription>
          可选，默认关闭。开启后，error 及以上的错误会自动推送到所选通道（warn/info 仍需要手动点「发送到远端」）。
          上传前会脱敏：Authorization / API key、用户主目录与盘符、邮箱与 IP 地址都会被替换。
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1">
              通道
            </div>
            <Select
              value={draft.error_upstream_channel}
              onValueChange={(v) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_channel: v as UpstreamChannel,
                }))
              }
            >
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">关闭(仅本地)</SelectItem>
                <SelectItem value="gitea">Gitea Issues (git.galiais.com 默认)</SelectItem>
                <SelectItem value="gitlab">GitLab Issues</SelectItem>
                <SelectItem value="webhook">Webhook</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground mb-1">
              自动发送阈值
            </div>
            <Select
              value={draft.error_upstream_auto_severity}
              onValueChange={(v) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_auto_severity:
                    v as SettingsState["error_upstream_auto_severity"],
                }))
              }
            >
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">全部手动</SelectItem>
                <SelectItem value="error">error 及以上自动发送</SelectItem>
                <SelectItem value="all">全部自动发送</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {(draft.error_upstream_channel === "gitlab" ||
          draft.error_upstream_channel === "gitea") && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Input
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "Gitea Base URL  https://git.galiais.com"
                  : "GitLab Base URL  https://gitlab.example.com"
              }
              value={draft.error_upstream_gitlab_base_url}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_base_url: e.target.value,
                }))
              }
            />
            <Input
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "项目路径  Shiro/LoraHubReport"
                  : "项目路径  group/project"
              }
              value={draft.error_upstream_gitlab_repo}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_repo: e.target.value,
                }))
              }
            />
            <Input
              type="password"
              placeholder={
                draft.error_upstream_channel === "gitea"
                  ? "Gitea Personal Access Token (write:issue scope)"
                  : "GitLab Personal Access Token (api scope)"
              }
              value={draft.error_upstream_gitlab_token}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_gitlab_token: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
            <p className="md:col-span-2 text-[11px] text-muted-foreground">
              提示:留空则回退到环境变量(
              {draft.error_upstream_channel === "gitea"
                ? "LORAHUB_GITEA_TOKEN"
                : "LORAHUB_GITLAB_TOKEN"}
              ,或通用 LORAHUB_REPORT_TOKEN)。这样 settings.json 不会留下明文 token。
            </p>
          </div>
        )}
        {draft.error_upstream_channel === "webhook" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Input
              placeholder="Webhook URL  https://hooks.example.com/lorahub"
              value={draft.error_upstream_webhook_url}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_webhook_url: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
            <Input
              type="password"
              placeholder="Authorization 头（可选，如 Bearer xxx）"
              value={draft.error_upstream_webhook_auth_header}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  error_upstream_webhook_auth_header: e.target.value,
                }))
              }
              className="md:col-span-2"
            />
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <Button
            size="sm"
            variant="outline"
            onClick={onProbe}
            disabled={probing || draft.error_upstream_channel === "off" || !cfg}
            className="gap-1.5"
          >
            <Network className="size-3" />
            {probing ? "测试中…" : "测试连通"}
          </Button>
          <Button
            size="sm"
            onClick={onSave}
            disabled={saving || !cfg}
            className="gap-1.5"
          >
            {saving ? "保存中…" : "保存"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
