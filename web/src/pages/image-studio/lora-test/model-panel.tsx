/**
 * 左侧模型面板 — 选择训练任务与 checkpoint。
 */
import type { LoraTestJob } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Field } from "./fields"

export function ModelPanel({
  loading,
  jobs,
  jobId,
  checkpointPath,
  onJobChange,
  onCheckpointChange,
  selectedJob,
}: {
  loading: boolean
  jobs: LoraTestJob[]
  jobId: string
  checkpointPath: string
  onJobChange: (value: string) => void
  onCheckpointChange: (value: string) => void
  selectedJob: LoraTestJob | null
}) {
  if (loading) return <Skeleton className="h-48" />
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>模型</CardTitle>
        <CardDescription>从训练产物中选择 LoRA checkpoint。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Field label="训练任务">
          <Select value={jobId} onValueChange={(value) => value && onJobChange(value)}>
            <SelectTrigger>
              <SelectValue placeholder="选择任务" />
            </SelectTrigger>
            <SelectContent>
              {jobs.map((job) => (
                <SelectItem key={job.job_id} value={job.job_id}>
                  {job.output_name ?? job.job_id.slice(-8)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Checkpoint">
          <Select
            value={checkpointPath}
            onValueChange={(value) => value && onCheckpointChange(value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="选择 LoRA" />
            </SelectTrigger>
            <SelectContent>
              {(selectedJob?.checkpoints ?? []).map((ckpt) => (
                <SelectItem key={ckpt.path} value={ckpt.path}>
                  {ckpt.path}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        {selectedJob ? (
          <div className="rounded-[6px] border border-border/60 bg-muted/30 p-2 text-[11px] text-muted-foreground">
            <div className="font-mono text-foreground">{selectedJob.backend ?? "unknown"}</div>
            <div className="mt-1 break-all">{selectedJob.workspace}</div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">没有可测试的 LoRA 产物。</p>
        )}
      </CardContent>
    </Card>
  )
}
