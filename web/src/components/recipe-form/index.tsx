/**
 * Visual recipe editor — covers every field that influences training.
 *
 * Built directly against RecipeConfig (lorahub/core/config/schema.py) so each
 * widget knows its semantics; the form is collapsible per section, validates
 * locally, and surfaces server validation errors next to the offending field.
 */
import { useCallback, useMemo } from "react"
import {
  Cpu,
  FileImage,
  Folder,
  Image,
  Layers,
  PaintBucket,
  Settings2,
  SlidersHorizontal,
  Wand2,
} from "lucide-react"
import type { ValidationFieldError } from "@/lib/api"
import { BaseModelFields } from "./sections/base-model"
import { DatasetFields } from "./sections/dataset"
import { NetworkFields } from "./sections/network"
import { OptimizerFields } from "./sections/optimizer"
import { ScheduleFields } from "./sections/schedule"
import { PrecisionFields } from "./sections/precision"
import { SamplingFields } from "./sections/sampling"
import { OutputFields } from "./sections/output"
import { BackendFields } from "./sections/backend"
import { buildErrorMap, setIn } from "./types"
import type { RecipeFormValue } from "./types"
import { Section } from "./widgets"

export type { RecipeFormValue } from "./types"

interface RecipeFormProps {
  value: RecipeFormValue
  onChange: (next: RecipeFormValue) => void
  errors?: ValidationFieldError[]
}

export function RecipeForm({ value, onChange, errors }: RecipeFormProps) {
  const errorMap = useMemo(() => buildErrorMap(errors), [errors])

  // Stable updater factory to keep child callbacks identity-stable when
  // value/onChange themselves are stable.
  const set = useCallback(
    (path: ReadonlyArray<string | number>, next: unknown) => {
      onChange(setIn(value, path, next))
    },
    [value, onChange],
  )

  return (
    <div className="space-y-3">
      <Section
        icon={<Cpu className="size-3.5" />}
        title="Base model"
        subtitle="Architecture and checkpoint to fine-tune"
        defaultOpen
      >
        <BaseModelFields value={value.base_model} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<FileImage className="size-3.5" />}
        title="Dataset"
        subtitle="Where the training images live and how they're loaded"
        defaultOpen
      >
        <DatasetFields value={value.dataset} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Layers className="size-3.5" />}
        title="Network"
        subtitle="LoRA structure: rank, alpha, target modules"
      >
        <NetworkFields value={value.network} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<SlidersHorizontal className="size-3.5" />}
        title="Optimizer & learning rate"
        subtitle="How weights move during training"
      >
        <OptimizerFields value={value.optimizer} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Settings2 className="size-3.5" />}
        title="Schedule"
        subtitle="Epochs, batch size, gradient accumulation"
        defaultOpen
      >
        <ScheduleFields value={value.schedule} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<PaintBucket className="size-3.5" />}
        title="Precision & memory"
        subtitle="Mixed precision, gradient checkpointing, latent cache"
      >
        <PrecisionFields value={value} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Image className="size-3.5" />}
        title="Sampling"
        subtitle="Generate preview images during training"
      >
        <SamplingFields value={value.sampling} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Folder className="size-3.5" />}
        title="Output"
        subtitle="Filename, save cadence, dtype"
      >
        <OutputFields value={value.output} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Wand2 className="size-3.5" />}
        title="Backend"
        subtitle="Override the kohya checkout / Python on a per-recipe basis"
      >
        <BackendFields value={value.backend} set={set} errorMap={errorMap} />
      </Section>
    </div>
  )
}
