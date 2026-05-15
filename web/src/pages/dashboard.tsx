import { useQuery } from "@tanstack/react-query"
import { Activity, CircleCheck, CircleX, Loader2, Pause } from "lucide-react"
import { api, type JobSummary } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export function DashboardPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 5000 })
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.listJobs, refetchInterval: 3000 })

  const all = jobs.data?.jobs ?? []
  const running = all.filter((j) => j.state === "running").length
  const succeeded = all.filter((j) => j.state === "succeeded").length
  const failed = all.filter(
    (j) => j.state === "failed" || j.state === "interrupted" || j.state === "canceled",
  ).length

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 max-w-[1100px]">
      <header className="space-y-1">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          Overview
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Live status of the LoraHub training workbench.
        </p>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          icon={<Activity className="size-3.5" />}
          label="Server"
          value={health.isError ? "down" : (health.data?.status ?? "...")}
          tone={health.isError ? "destructive" : "default"}
        />
        <StatCard
          icon={<Loader2 className="size-3.5" />}
          label="Running"
          value={running.toString()}
          tone="primary"
        />
        <StatCard
          icon={<CircleCheck className="size-3.5" />}
          label="Succeeded"
          value={succeeded.toString()}
        />
        <StatCard
          icon={<CircleX className="size-3.5" />}
          label="Failed"
          value={failed.toString()}
          tone={failed > 0 ? "warning" : "default"}
        />
      </div>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Recent jobs</CardTitle>
          <CardDescription>Latest training runs from this workspace.</CardDescription>
        </CardHeader>
        <CardContent>
          {all.length === 0 ? (
            <EmptyState />
          ) : (
            <ul className="divide-y divide-border/50">
              {all.slice(-8).reverse().map((j) => (
                <RecentRow key={j.id} job={j} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ReactNode
  label: string
  value: string
  tone?: "default" | "primary" | "destructive" | "warning"
}) {
  const toneStyle = {
    default: "text-foreground",
    primary: "text-primary",
    destructive: "text-destructive",
    warning: "text-amber-700 dark:text-amber-400",
  }[tone]
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardContent className="px-4 py-3">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          {icon}
          {label}
        </div>
        <div className={`mt-1.5 text-2xl font-semibold tracking-tight tabular-nums ${toneStyle}`}>
          {value}
        </div>
      </CardContent>
    </Card>
  )
}

function RecentRow({ job }: { job: JobSummary }) {
  return (
    <li className="py-2.5 flex items-center gap-3 text-sm">
      <StateBadge state={job.state} />
      <code className="text-[11px] text-muted-foreground font-mono">{job.id.slice(-8)}</code>
      <span className="ml-auto text-xs text-muted-foreground">
        {new Date(job.created_at).toLocaleString()}
      </span>
    </li>
  )
}

function EmptyState() {
  return (
    <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center">
      <Pause className="size-5 mx-auto text-muted-foreground/60" />
      <div className="mt-2 text-sm font-medium">No jobs yet</div>
      <div className="text-xs text-muted-foreground">
        Run <code className="text-foreground">lorahub train recipe.yaml</code> or POST to{" "}
        <code>/jobs</code> to start one.
      </div>
    </div>
  )
}

export function StateBadge({ state }: { state: string }) {
  const variant = {
    running: "default",
    succeeded: "secondary",
    failed: "destructive",
    canceled: "outline",
    canceling: "outline",
    queued: "outline",
    interrupted: "destructive",
  }[state] as "default" | "secondary" | "destructive" | "outline" | undefined

  return (
    <Badge variant={variant ?? "outline"} className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]">
      {state}
    </Badge>
  )
}
