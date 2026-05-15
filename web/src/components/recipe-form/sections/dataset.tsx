import { memo } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { CAPTION_STRATEGY_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, IntInput, PathInput, ResolutionInput, Row } from "../widgets"

export const DatasetFields = memo(function DatasetFields({
  value,
  set,
  errorMap,
}: {
  value: RecipeFormValue["dataset"]
  set: Setter
  errorMap: ErrorMap
}) {
  const bucket = value.bucket ?? {}
  const caption = value.caption ?? {}
  return (
    <>
      <Row label="Source" required errors={errorMap.get("dataset.source")}>
        <PathInput
          value={value.source}
          onChange={(v) => set(["dataset", "source"], v)}
          placeholder="./datasets/my_character"
        />
      </Row>
      <Row label="Resolution" errors={errorMap.get("dataset.resolution")}>
        <ResolutionInput
          value={value.resolution}
          onChange={(v) => set(["dataset", "resolution"], v)}
        />
      </Row>
      <Row label="Repeats" description="How many times each image is seen per epoch.">
        <IntInput
          min={1}
          value={value.num_repeats}
          onChange={(v) => set(["dataset", "num_repeats"], v ?? 1)}
        />
      </Row>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
            Bucket
          </span>
          <Switch
            checked={bucket.enabled ?? true}
            onCheckedChange={(v) => set(["dataset", "bucket", "enabled"], v)}
          />
        </div>
        {bucket.enabled !== false && (
          <div className="grid grid-cols-3 gap-3">
            <div>
              <Label className="text-[11px] text-muted-foreground">min</Label>
              <IntInput
                min={64}
                value={bucket.min ?? 256}
                onChange={(v) => set(["dataset", "bucket", "min"], v ?? 256)}
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">max</Label>
              <IntInput
                min={64}
                value={bucket.max ?? 2048}
                onChange={(v) => set(["dataset", "bucket", "max"], v ?? 2048)}
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">step</Label>
              <IntInput
                min={8}
                value={bucket.step ?? 64}
                onChange={(v) => set(["dataset", "bucket", "step"], v ?? 64)}
              />
            </div>
          </div>
        )}
      </div>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          Caption
        </div>
        <Row label="Strategy">
          <EnumSelect
            value={caption.strategy ?? "tag_file"}
            onChange={(v) => set(["dataset", "caption", "strategy"], v)}
            options={CAPTION_STRATEGY_OPTIONS}
          />
        </Row>
        <Row label="Extension" description=".txt is the kohya default.">
          <Input
            value={caption.ext ?? ".txt"}
            className="font-mono w-32"
            onChange={(e) => set(["dataset", "caption", "ext"], e.target.value)}
          />
        </Row>
        <Row label="Shuffle tags" description="Randomize comma-separated tags each step.">
          <Switch
            checked={caption.shuffle ?? true}
            onCheckedChange={(v) => set(["dataset", "caption", "shuffle"], v)}
          />
        </Row>
        <Row
          label="Drop rate"
          description="Probability (0-1) that a tag is dropped per step."
          errors={errorMap.get("dataset.caption.drop_rate")}
        >
          <FloatInput
            step={0.05}
            value={caption.drop_rate ?? 0}
            onChange={(v) => set(["dataset", "caption", "drop_rate"], v ?? 0)}
          />
        </Row>
      </div>
    </>
  )
})
