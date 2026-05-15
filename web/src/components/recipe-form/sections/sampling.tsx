import { memo } from "react"
import { Switch } from "@/components/ui/switch"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { IntInput, PathInput, ResolutionInput, Row } from "../widgets"

export const SamplingFields = memo(function SamplingFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["sampling"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const enabled = v.enabled ?? true
  return (
    <>
      <Row label="Enabled" description="Render preview images during training.">
        <Switch checked={enabled} onCheckedChange={(b) => set(["sampling", "enabled"], b)} />
      </Row>
      {enabled && (
        <>
          <Row label="Every N epochs">
            <IntInput
              min={1}
              value={v.every_n_epochs ?? 1}
              onChange={(n) => set(["sampling", "every_n_epochs"], n ?? 1)}
            />
          </Row>
          <Row label="Prompts file" description="Plain text file, one prompt per line." errors={errorMap.get("sampling.prompts_file")}>
            <PathInput
              value={v.prompts_file ?? ""}
              onChange={(s) => set(["sampling", "prompts_file"], s || null)}
              placeholder="./prompts.txt"
            />
          </Row>
          <Row label="Resolution">
            <ResolutionInput
              value={v.resolution}
              onChange={(r) => set(["sampling", "resolution"], r)}
            />
          </Row>
          <Row label="Seed">
            <IntInput value={v.seed ?? 42} onChange={(n) => set(["sampling", "seed"], n ?? 42)} />
          </Row>
        </>
      )}
    </>
  )
})
