/**
 * Shared types and pure update helpers for the config form.
 *
 * Kept in a dedicated module so each section file can import only the symbols
 * it needs without dragging in widgets or option lists.
 */
import type { ValidationFieldError } from "@/lib/api"

export interface ConfigFormValue {
  schema_version?: string
  base_model: {
    arch: string
    // SDXL sub-variant; only meaningful when arch === "sdxl". Empty string
    // (or absent) means「无变体」.
    arch_variant?: string
    checkpoint: string
    vae?: string | null
  }
  dataset: {
    source: string
    resolution: [number, number] | number[]
    bucket?: {
      enabled?: boolean
      min?: number
      max?: number
      step?: number
    }
    caption?: {
      strategy?: string
      ext?: string
      shuffle?: boolean
      drop_rate?: number
    }
    num_repeats?: number
    val_split?: number
  }
  network?: {
    type?: string
    rank?: number
    alpha?: number
    target_unet?: boolean
    target_text_encoder?: boolean
    conv_dim?: number | null
    conv_alpha?: number | null
    network_dropout?: number
    rank_dropout?: number
    module_dropout?: number
    scale_weight_norms?: number | null
  }
  optimizer?: {
    type?: string
    lr?: { unet?: number; text_encoder?: number }
    schedule?: string
    warmup_steps?: number
    betas?: [number, number] | number[]
    weight_decay?: number
    eps?: number
    optimizer_args?: Record<string, string>
  }
  loss?: {
    min_snr_gamma?: number | null
    noise_offset?: number
    ip_noise_gamma?: number | null
    prior_loss_weight?: number
    loss_type?: string
    debiased_estimation?: boolean
    masked_loss?: boolean
    scale_v_pred_loss_like_noise_pred?: boolean
    v_parameterization?: boolean
  }
  schedule?: {
    epochs?: number
    batch_size?: number
    grad_accum?: number
    max_steps?: number | null
  }
  precision?: string
  gradient_checkpointing?: boolean
  cache_latents?: boolean
  sampling?: {
    enabled?: boolean
    every_n_epochs?: number
    prompts_file?: string | null
    resolution?: [number, number] | number[]
    seed?: number
    // Attention backend reserved for the sample/validation path. Mirrors
    // SamplingConfig.attention in lorahub/core/config/schema.py. Today both
    // backends emit a warning and skip wiring this through (training stays
    // on its own attention kernel so SageAttention's missing backward
    // can't poison gradients); the field is preserved so authored recipes
    // stay valid once the wrapper lands.
    attention?: string
  }
  output?: {
    name?: string
    save_every_n_epochs?: number
    save_dtype?: string
    output_dir?: string | null
  }
  backend?: {
    type?: string
    pin_version?: string | null
    sd_scripts_path?: string | null
    python_executable?: string | null
    extra_args?: Record<string, unknown>
    diffusion_pipe?: {
      pipeline_stages?: number
      gradient_clipping?: number
      partition_method?: string
      caching_batch_size?: number
      steps_per_print?: number
      blocks_to_swap?: number
      compile?: boolean
      eval_every_n_epochs?: number | null
      eval_before_first_step?: boolean
      eval_micro_batch_size_per_gpu?: number
      enable_wandb?: boolean
      tracker_name?: string | null
      run_name?: string | null
      min_ar?: number
      max_ar?: number
      num_ar_buckets?: number
      cache_shuffle_num?: number
      skip_empty_caption?: boolean
      model_paths?: Record<string, string>
    }
  }
  resume?: {
    save_state?: boolean
    save_state_at_end?: boolean
    save_state_every_n_epochs?: number | null
  }
  validation?: {
    every_n_epochs?: number
    max_samples?: number | null
  }
  optimization?: {
    torch_compile?: boolean
    fused_backward_pass?: boolean
    full_bf16?: boolean
    blocks_to_swap?: number
  attention?: {
    training?: string
    split?: boolean
  }
  [k: string]: unknown
}

export type ErrorMap = Map<string, string[]>
export type Setter = (path: ReadonlyArray<string | number>, next: unknown) => void

// ------------------------------------------------------ pure update helpers

export function setIn<T extends object>(
  obj: T,
  path: ReadonlyArray<string | number>,
  value: unknown,
): T {
  if (path.length === 0) return value as T
  const cloned: any = Array.isArray(obj) ? [...(obj as any)] : { ...(obj as any) }
  const [head, ...rest] = path
  cloned[head as any] = setIn(
    cloned[head as any] ?? (typeof rest[0] === "number" ? [] : {}),
    rest,
    value,
  )
  return cloned
}

export function buildErrorMap(errors: ValidationFieldError[] | undefined): ErrorMap {
  const m = new Map<string, string[]>()
  if (!errors) return m
  for (const e of errors) {
    const key = e.loc.join(".")
    const arr = m.get(key) ?? []
    arr.push(e.msg)
    m.set(key, arr)
  }
  return m
}
