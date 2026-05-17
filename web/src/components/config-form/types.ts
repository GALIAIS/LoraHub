/**
 * Shared types and pure update helpers for the config form.
 *
 * Kept in a dedicated module so each section file can import only the symbols
 * it needs without dragging in widgets or option lists.
 *
 * The shape mirrors `lorahub/core/config/schema.py` (TrainingConfig + nested
 * pydantic models). Every field that the python schema accepts has a matching
 * (optional) entry here so the form can persist untouched fields verbatim
 * across save / reload roundtrips.
 */
import type { ValidationFieldError } from "@/lib/api"

export interface ArchPathsValue {
  // FLUX / SD3 / FLUX2 component checkpoints
  clip_l?: string | null
  clip_g?: string | null
  t5xxl?: string | null
  ae?: string | null
  // Generic (Anima / Wan / HunyuanImage / chroma)
  transformer?: string | null
  text_encoder?: string | null
  llm?: string | null
  byt5?: string | null
  // Anima-specific
  qwen3?: string | null
  t5_tokenizer?: string | null
  llm_adapter?: string | null
  // Token length caps
  t5xxl_max_token_length?: number | null
  qwen3_max_token_length?: number | null
  t5_max_token_length?: number | null
  // Attention masking + dropout
  apply_t5_attn_mask?: boolean
  apply_lg_attn_mask?: boolean
  t5_dropout_rate?: number
  clip_l_dropout_rate?: number
  clip_g_dropout_rate?: number
  // SD3 positional-embed crop
  pos_emb_random_crop_rate?: number
  enable_scaled_pos_embed?: boolean
  // FLUX dev distilled guidance
  guidance_scale?: number | null
  // TE device / dtype
  t5xxl_device?: string | null
  t5xxl_dtype?: string | null
  // VAE / TE memory tweaks
  vae_chunk_size?: number | null
  vae_disable_cache?: boolean
  text_encoder_cpu?: boolean
}

export interface PerModuleLRValue {
  llm_adapter?: number | null
  self_attn?: number | null
  cross_attn?: number | null
  mlp?: number | null
  mod?: number | null
}

export interface DatasetSubsetValue {
  path?: string
  num_repeats?: number
  mask_path?: string | null
  ar_buckets?: number[] | null
  caption_prefix?: string | null
}

export interface ConfigFormValue {
  schema_version?: string
  base_model: {
    arch: string
    arch_variant?: string
    checkpoint: string
    vae?: string | null
    arch_paths?: ArchPathsValue
  }
  dataset: {
    source: string
    resolution: [number, number] | number[]
    bucket?: {
      enabled?: boolean
      min?: number
      max?: number
      step?: number
      no_upscale?: boolean
      skip_image_resolution?: boolean
      resize_interpolation?: string | null
      ar_buckets?: number[] | null
    }
    caption?: {
      strategy?: string
      ext?: string
      shuffle?: boolean
      drop_rate?: number
      dropout_every_n_epochs?: number
      tag_dropout_rate?: number
      keep_tokens?: number
      keep_tokens_separator?: string | null
      secondary_separator?: string | null
      enable_wildcard?: boolean
      prefix?: string | null
      suffix?: string | null
      max_token_length?: number | null
      token_warmup_min?: number | null
      token_warmup_step?: number | null
      weighted?: boolean
      shuffle_delimiter?: string | null
      shuffle_tags?: boolean
    }
    num_repeats?: number
    val_split?: number
    subsets?: DatasetSubsetValue[]
    frame_buckets?: number[]
    conditioning_dir?: string | null
    reg_source?: string | null
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
    init_from?: string | null
    dim_from_weights?: string | null
    base_weights?: string[]
    base_weights_multiplier?: number[]
    fuse_adapters?: Array<Record<string, unknown>>
    module_lr?: PerModuleLRValue | null
    dtype?: string | null
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
    max_grad_norm?: number
    scheduler_module?: string | null
    scheduler_args?: Record<string, string>
    scheduler_num_cycles?: number
    scheduler_power?: number
    scheduler_timescale?: number | null
    scheduler_min_lr_ratio?: number | null
    gradient_release?: boolean
  }
  loss?: {
    min_snr_gamma?: number | null
    noise_offset?: number
    noise_offset_random_strength?: boolean
    multires_noise_iterations?: number | null
    multires_noise_discount?: number
    adaptive_noise_scale?: number | null
    ip_noise_gamma?: number | null
    ip_noise_gamma_random_strength?: boolean
    zero_terminal_snr?: boolean
    min_timestep?: number | null
    max_timestep?: number | null
    prior_loss_weight?: number
    loss_type?: string
    huber_schedule?: string | null
    huber_c?: number | null
    huber_scale?: number | null
    debiased_estimation?: boolean
    masked_loss?: boolean
    scale_v_pred_loss_like_noise_pred?: boolean
    v_parameterization?: boolean
    v_pred_like_loss?: number | null
    pseudo_huber_c?: number | null
  }
  flow_match?: {
    timestep_sampling?: string | null
    sigmoid_scale?: number | null
    model_prediction_type?: string | null
    discrete_flow_shift?: number | null
    training_shift?: number | null
    weighting_scheme?: string | null
    logit_mean?: number | null
    logit_std?: number | null
    mode_scale?: number | null
  }
  schedule?: {
    epochs?: number
    batch_size?: number
    grad_accum?: number
    max_steps?: number | null
    seed?: number | null
    lr_decay_steps?: number | null
  }
  precision?: string
  gradient_checkpointing?: boolean
  cache_latents?: boolean
  cache_latents_to_disk?: boolean
  skip_cache_check?: boolean
  cache_info?: boolean
  train_inpainting?: boolean
  sampling?: {
    enabled?: boolean
    every_n_epochs?: number
    every_n_steps?: number | null
    at_first?: boolean
    prompts_file?: string | null
    resolution?: [number, number] | number[]
    seed?: number
    attention?: string
  }
  output?: {
    name?: string
    save_every_n_epochs?: number
    save_every_n_steps?: number | null
    save_every_n_examples?: number | null
    save_last_n_epochs?: number | null
    save_last_n_steps?: number | null
    save_dtype?: string
    output_dir?: string | null
    training_comment?: string | null
    no_metadata?: boolean
    metadata?: Record<string, string>
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
      partition_split?: number[] | null
      caching_batch_size?: number
      steps_per_print?: number
      blocks_to_swap?: number
      compile?: boolean
      reentrant_activation_checkpointing?: boolean
      disable_block_swap_for_eval?: boolean
      image_micro_batch_size_per_gpu?: number | null
      image_eval_micro_batch_size_per_gpu?: number | null
      eval_gradient_accumulation_steps?: number
      eval_every_n_epochs?: number | null
      eval_every_n_steps?: number | null
      eval_every_n_examples?: number | null
      eval_before_first_step?: boolean
      eval_micro_batch_size_per_gpu?: number
      checkpoint_every_n_epochs?: number | null
      checkpoint_every_n_minutes?: number | null
      force_constant_lr?: number | null
      uncond_fraction?: number
      x_axis_examples?: boolean
      logging_steps?: number
      transformer_dtype?: string | null
      diffusion_model_dtype?: string | null
      timestep_sample_method?: string | null
      eval_datasets?: Array<Record<string, string>>
      video_clip_mode?: string
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
    resume_from?: string | null
    save_last_n_epochs_state?: number | null
    save_last_n_steps_state?: number | null
    skip_until_initial_step?: boolean
    initial_epoch?: number | null
    initial_step?: number | null
  }
  validation?: {
    every_n_epochs?: number
    every_n_steps?: number | null
    max_samples?: number | null
    seed?: number | null
  }
  optimization?: {
    torch_compile?: boolean
    fused_backward_pass?: boolean
    full_bf16?: boolean
    full_fp16?: boolean
    blocks_to_swap?: number
    fp8_base?: boolean
    fp8_base_unet?: boolean
    fp8_scaled?: boolean
    fp8_vl_text_encoder?: boolean
    lowram?: boolean
    highvram?: boolean
    no_half_vae?: boolean
    disable_mmap_load_safetensors?: boolean
    cpu_offload_checkpointing?: boolean
    unsloth_offload_checkpointing?: boolean
    cache_text_encoder_outputs?: boolean
    cache_text_encoder_outputs_to_disk?: boolean
  }
  attention?: {
    training?: string
    split?: boolean
  }
  dataloader?: {
    num_workers?: number
    persistent_workers?: boolean
    vae_batch_size?: number
    text_encoder_batch_size?: number | null
    cache_shuffle_num?: number
    map_num_proc?: number | null
  }
  augmentation?: {
    flip?: boolean
    color?: boolean
    random_crop?: boolean
    face_crop_aug_range?: string | null
    alpha_mask?: boolean
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
