import { memo } from "react"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, Row, TextInput, ToggleSwitch } from "../widgets"

/**
 * Editor for the top-level [monitoring] section. Strict mirror of
 * ``MonitoringConfig`` in ``lorahub/core/config/schema.py`` and
 * ``wandb.init()`` per the official docs:
 *   - project / entity / runName / runId / group / jobType -> wandb.init kwargs
 *   - tags / notes / mode / resume                         -> wandb.init kwargs
 *   - baseUrl                                              -> WANDB_BASE_URL
 *
 * Identity transports:
 *   - kohya  : --log_with / --log_tracker_name / --wandb_run_name
 *   - anima  : log_with / log_tracker_name / wandb_run_name in TOML
 *   - dp     : [monitoring] enable_wandb / wandb_tracker_name / wandb_run_name
 *   - All extra fields (entity, tags, notes, run_id, group, job_type, mode,
 *     resume, base_url) ride WANDB_* env vars from
 *     ``lorahub.api.wandb_env.wandb_env``.
 *
 * The api key lives in user settings (Settings → 网络) and is forwarded as
 * ``WANDB_API_KEY`` so secrets never touch the config YAML.
 */
export const MonitoringFields = memo(function MonitoringFields({
  value,
  set,
  errorMap: _errorMap,
}: {
  value: ConfigFormValue["monitoring"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const enabled = v.enableWandb ?? false
  const tagsText = (v.tags ?? []).join(",")

  return (
    <>
      <Row label="启用 W&amp;B" description="将训练指标推送到 wandb.ai 或自托管 W&amp;B Server。">
        <ToggleSwitch
          checked={enabled}
          onCheckedChange={(b) => set(["monitoring", "enableWandb"], b)}
        />
      </Row>
      {enabled && (
        <>
          <Row
            label="Project"
            description="wandb.init(project=...);留空使用账号默认 project。"
          >
            <TextInput
              className="w-64"
              value={v.project ?? ""}
              onChange={(s) => set(["monitoring", "project"], s || null)}
              placeholder="my-lora-runs"
            />
          </Row>
          <Row
            label="Entity"
            description="wandb.init(entity=...);user 或 team,留空走默认 entity。"
          >
            <TextInput
              className="w-64"
              value={v.entity ?? ""}
              onChange={(s) => set(["monitoring", "entity"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row label="Run 名称" description="wandb.init(name=...);留空 wandb 自动生成。">
            <TextInput
              className="w-64"
              value={v.runName ?? ""}
              onChange={(s) => set(["monitoring", "runName"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="Run ID"
            description="wandb.init(id=...);用于断点续训,留空 wandb 自动分配。"
          >
            <TextInput
              className="w-64"
              value={v.runId ?? ""}
              onChange={(s) => set(["monitoring", "runId"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row label="Group" description="wandb.init(group=...);用于分组多次实验。">
            <TextInput
              className="w-64"
              value={v.group ?? ""}
              onChange={(s) => set(["monitoring", "group"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="Job 类型"
            description="wandb.init(job_type=...);常见值如 train / eval。"
          >
            <TextInput
              className="w-64"
              value={v.jobType ?? ""}
              onChange={(s) => set(["monitoring", "jobType"], s || null)}
              placeholder="train"
            />
          </Row>
          <Row label="Tags" description="逗号分隔。映射到 wandb.init(tags=[...]).">
            <TextInput
              className="w-64"
              value={tagsText}
              onChange={(s) => {
                const arr = s
                  .split(",")
                  .map((t) => t.trim())
                  .filter((t) => t.length > 0)
                set(["monitoring", "tags"], arr)
              }}
              placeholder="sdxl,exp"
            />
          </Row>
          <Row label="Notes" description="wandb.init(notes=...);Markdown 描述。">
            <TextInput
              className="w-64"
              value={v.notes ?? ""}
              onChange={(s) => set(["monitoring", "notes"], s || null)}
              placeholder="（可选）"
            />
          </Row>
          <Row
            label="Mode"
            description="online=实时同步;offline=本地缓冲后 wandb sync;disabled=完全关闭。"
          >
            <EnumSelect
              value={v.mode ?? "online"}
              options={[
                { value: "online", label: "online" },
                { value: "offline", label: "offline" },
                { value: "disabled", label: "disabled" },
                { value: "shared", label: "shared" },
              ]}
              onChange={(s) =>
                set(
                  ["monitoring", "mode"],
                  s === "online" ? null : (s as "offline" | "disabled" | "shared"),
                )
              }
            />
          </Row>
          <Row label="Resume" description="断点续训策略,需要配合 Run ID。">
            <EnumSelect
              value={v.resume ?? "never"}
              options={[
                { value: "never", label: "never" },
                { value: "allow", label: "allow" },
                { value: "must", label: "must" },
                { value: "auto", label: "auto" },
              ]}
              onChange={(s) =>
                set(
                  ["monitoring", "resume"],
                  s === "never" ? null : (s as "allow" | "must" | "auto"),
                )
              }
            />
          </Row>
          <Row
            label="Base URL"
            description="自托管 W&amp;B Server 地址,留空使用 wandb.ai。"
          >
            <TextInput
              className="w-72"
              value={v.baseUrl ?? ""}
              onChange={(s) => set(["monitoring", "baseUrl"], s || null)}
              placeholder="https://wandb.your-domain.com"
            />
          </Row>
          <p className="text-[11px] text-muted-foreground px-1 leading-relaxed">
            API Key 在「设置 → 网络」中填写,启动训练时会以
            <code> WANDB_API_KEY </code>注入子进程,不会写入 config。
          </p>
        </>
      )}
    </>
  )
})
