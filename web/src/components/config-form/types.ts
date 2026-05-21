/**
 * Shared types and pure update helpers for the config form.
 *
 * Kept in a dedicated module so each section file can import only the symbols
 * it needs without dragging in widgets or option lists.
 *
 * The shape mirrors `lorahub/core/config/schema.py` (TrainingConfig + nested
 * pydantic models). Field names are camelCase to match the YAML wire format
 * after the schema-wide alias_generator=to_camel migration; the Python schema
 * still accepts both snake_case and camelCase, so server-side input never
 * breaks even if a few callers haven't migrated yet.
 */
import type { ValidationFieldError } from "@/lib/api"

export interface ArchPathsValue {
  // FLUX / SD3 / FLUX2 component checkpoints
  clipL?: string | null
  clipG?: string | null
  t5xxl?: string | null
  ae?: string | null
  // Generic (Anima / Wan / HunyuanImage / chroma)
  transformer?: string | null
  textEncoder?: string | null
  llm?: string | null
  byt5?: string | null
  // Anima-specific
  qwen3?: string | null
  t5Tokenizer?: string | null
  llmAdapter?: string | null
  // Token length caps
  t5xxlMaxTokenLength?: number | null
  qwen3MaxTokenLength?: number | null
  t5MaxTokenLength?: number | null
  // Attention masking + dropout
  applyT5AttnMask?: boolean
  applyLgAttnMask?: boolean
  t5DropoutRate?: number
  clipLDropoutRate?: number
  clipGDropoutRate?: number
  // SD3 positional-embed crop
  posEmbRandomCropRate?: number
  enableScaledPosEmbed?: boolean
  // FLUX dev distilled guidance
  guidanceScale?: number | null
  // TE device / dtype
  t5xxlDevice?: string | null
  t5xxlDtype?: string | null
  // VAE / TE memory tweaks
  vaeChunkSize?: number | null
  vaeDisableCache?: boolean
  textEncoderCpu?: boolean
}

export interface PerModuleLRValue {
  llmAdapter?: number | null
  selfAttn?: number | null
  crossAttn?: number | null
  mlp?: number | null
  mod?: number | null
}

export interface DatasetSubsetValue {
  path?: string
  numRepeats?: number
  maskPath?: string | null
  arBuckets?: number[] | null
  captionPrefix?: string | null
}

export interface ConfigFormValue {
  schemaVersion?: string
  baseModel: {
    arch: string
    archVariant?: string
    checkpoint: string
    vae?: string | null
    archPaths?: ArchPathsValue
  }
  dataset: {
    source: string
    resolution: [number, number] | number[]
    bucket?: {
      enabled?: boolean
      min?: number
      max?: number
      step?: number
      noUpscale?: boolean
      skipImageResolution?: boolean
      resizeInterpolation?: string | null
      arBuckets?: number[] | null
    }
    caption?: {
      strategy?: string
      ext?: string
      shuffle?: boolean
      dropRate?: number
      dropoutEveryNEpochs?: number
      tagDropoutRate?: number
      keepTokens?: number
      /** Hard-drop list. Each entry is removed verbatim (case-insensitive
       *  substring match) from every caption at compile time, before the
       *  trainer reads them. Tag-style entries (``"1girl"``) and
       *  natural-language phrases (``"looking at viewer"``) both work. */
      dropTokens?: string[]
      keepTokensSeparator?: string | null
      secondarySeparator?: string | null
      enableWildcard?: boolean
      prefix?: string | null
      suffix?: string | null
      maxTokenLength?: number | null
      tokenWarmupMin?: number | null
      tokenWarmupStep?: number | null
      weighted?: boolean
      shuffleDelimiter?: string | null
      shuffleTags?: boolean
    }
    numRepeats?: number
    valSplit?: number
    subsets?: DatasetSubsetValue[]
    frameBuckets?: number[]
    conditioningDir?: string | null
    regSource?: string | null
  }
  network?: {
    type?: string
    rank?: number
    alpha?: number
    targetUnet?: boolean
    targetTextEncoder?: boolean
    convDim?: number | null
    convAlpha?: number | null
    networkDropout?: number
    rankDropout?: number
    moduleDropout?: number
    scaleWeightNorms?: number | null
    initFrom?: string | null
    dimFromWeights?: string | null
    baseWeights?: string[]
    baseWeightsMultiplier?: number[]
    fuseAdapters?: Array<Record<string, unknown>>
    moduleLr?: PerModuleLRValue | null
    dtype?: string | null
  }
  optimizer?: {
    type?: string
    lr?: { unet?: number; textEncoder?: number }
    schedule?: string
    warmupSteps?: number
    betas?: [number, number] | number[]
    weightDecay?: number
    eps?: number
    optimizerArgs?: Record<string, string>
    maxGradNorm?: number
    schedulerModule?: string | null
    schedulerArgs?: Record<string, string>
    schedulerNumCycles?: number
    schedulerPower?: number
    schedulerTimescale?: number | null
    schedulerMinLrRatio?: number | null
    gradientRelease?: boolean
  }
  loss?: {
    minSnrGamma?: number | null
    noiseOffset?: number
    noiseOffsetRandomStrength?: boolean
    multiresNoiseIterations?: number | null
    multiresNoiseDiscount?: number
    adaptiveNoiseScale?: number | null
    ipNoiseGamma?: number | null
    ipNoiseGammaRandomStrength?: boolean
    zeroTerminalSnr?: boolean
    minTimestep?: number | null
    maxTimestep?: number | null
    priorLossWeight?: number
    lossType?: string
    huberSchedule?: string | null
    huberC?: number | null
    huberScale?: number | null
    debiasedEstimation?: boolean
    maskedLoss?: boolean
    scaleVPredLossLikeNoisePred?: boolean
    vParameterization?: boolean
    vPredLikeLoss?: number | null
    pseudoHuberC?: number | null
  }
  flowMatch?: {
    timestepSampling?: string | null
    sigmoidScale?: number | null
    modelPredictionType?: string | null
    discreteFlowShift?: number | null
    trainingShift?: number | null
    weightingScheme?: string | null
    logitMean?: number | null
    logitStd?: number | null
    modeScale?: number | null
  }
  schedule?: {
    epochs?: number
    batchSize?: number
    gradAccum?: number
    maxSteps?: number | null
    seed?: number | null
    lrDecaySteps?: number | null
  }
  precision?: string
  gradientCheckpointing?: boolean
  cacheLatents?: boolean
  cacheLatentsToDisk?: boolean
  skipCacheCheck?: boolean
  cacheInfo?: boolean
  trainInpainting?: boolean
  sampling?: {
    enabled?: boolean
    everyNEpochs?: number
    everyNSteps?: number | null
    atFirst?: boolean
    promptsFile?: string | null
    resolution?: [number, number] | number[]
    seed?: number
  }
  output?: {
    name?: string
    saveEveryNEpochs?: number
    saveEveryNSteps?: number | null
    saveEveryNExamples?: number | null
    saveLastNEpochs?: number | null
    saveLastNSteps?: number | null
    saveDtype?: string
    outputDir?: string | null
    trainingComment?: string | null
    noMetadata?: boolean
    metadata?: Record<string, string>
  }
  backend?: {
    type?: string
    pinVersion?: string | null
    sdScriptsPath?: string | null  // legacy alias — accepted on load, no longer written
    repoPath?: string | null
    pythonExecutable?: string | null
    extraArgs?: Record<string, unknown>
    diffusionPipe?: {
      pipelineStages?: number
      gradientClipping?: number
      partitionMethod?: string
      partitionSplit?: number[] | null
      cachingBatchSize?: number
      stepsPerPrint?: number
      blocksToSwap?: number
      compile?: boolean
      reentrantActivationCheckpointing?: boolean
      disableBlockSwapForEval?: boolean
      imageMicroBatchSizePerGpu?: number | null
      imageEvalMicroBatchSizePerGpu?: number | null
      evalGradientAccumulationSteps?: number
      evalEveryNEpochs?: number | null
      evalEveryNSteps?: number | null
      evalEveryNExamples?: number | null
      evalBeforeFirstStep?: boolean
      evalMicroBatchSizePerGpu?: number
      checkpointEveryNEpochs?: number | null
      checkpointEveryNMinutes?: number | null
      forceConstantLr?: number | null
      uncondFraction?: number
      xAxisExamples?: boolean
      loggingSteps?: number
      transformerDtype?: string | null
      diffusionModelDtype?: string | null
      timestepSampleMethod?: string | null
      evalDatasets?: Array<Record<string, string>>
      videoClipMode?: string
      enableWandb?: boolean
      trackerName?: string | null
      runName?: string | null
      minAr?: number
      maxAr?: number
      numArBuckets?: number
      cacheShuffleNum?: number
      skipEmptyCaption?: boolean
      // NOTE: keys inside `modelPaths` are passed verbatim to diffusion-pipe
      // TOML, which expects literal snake_case (transformer_path / vae_path /
      // llm_path / ...). Do NOT rename these or dp won't recognise them.
      modelPaths?: Record<string, string>
    }
    /**
     * anima_lora-specific knobs. Mirrors `AnimaLoraOptions` in
     * `lorahub/core/config/schema.py`. Empty / undefined = upstream
     * defaults (lora.toml + presets.toml[default]).
     */
    animaLora?: {
      method?: "lora" | "postfix" | "chimera" | "easycontrol" | "ip_adapter"
      preset?:
        | "default"
        | "low_vram"
        | "graft"
        | "half"
        | "quarter"
        | "tenth"
        | "debug"
      outputName?: string
      networkModule?: string
      networkDim?: number
      networkAlpha?: number
      networkTrainUnetOnly?: boolean
      optimizerType?: "AdamW" | "AdamW8bit" | "Lion" | "Prodigy"
      lrScheduler?:
        | "constant"
        | "cosine"
        | "cosine_with_restarts"
        | "linear"
        | "polynomial"
      learningRate?: number
      maxTrainEpochs?: number
      saveEveryNEpochs?: number
      checkpointingEpochs?: number
      captionDropoutRate?: number
      timestepSampling?: "sigmoid" | "uniform" | "logit_normal"
      sigmoidScale?: number
      discreteFlowShift?: number
      weightingScheme?: "sigma_sqrt" | "logit_normal" | "mode" | "cosmap" | "min_snr_rf" | null
      minSnrGamma?: number | null
      logitMean?: number | null
      logitStd?: number | null
      modeScale?: number | null
      vrLossWeight?: number | null
      // ---- Training stabilisers ----
      ema?: boolean
      emaDecay?: number
      emaUseNumUpdates?: boolean
      nanGuard?: boolean
      nanGuardRecover?: boolean
      nanGuardMaxConsecutive?: number
      sampleGrid?: boolean
      cacheLatents?: boolean
      cacheLatentsToDisk?: boolean
      cacheTextEncoderOutputs?: boolean
      cacheTextEncoderOutputsToDisk?: boolean
      cacheLlmAdapterOutputs?: boolean
      useShuffledCaptionVariants?: boolean
      sampleRatio?: number | null
      staticTokenCount?: number
      vaeChunkSize?: number
      vaeDisableCache?: boolean
      noHalfVae?: boolean
      attnMode?: "flash" | "torch" | "flex" | "sageattn" | "xformers"
      xformers?: boolean
      splitAttn?: boolean
      compileMode?: "blocks" | "full" | null
      compileInductorMode?:
        | "default"
        | "reduce-overhead"
        | "max-autotune"
        | null
      useCustomDownAutograd?: boolean
      blocksToSwap?: number
      gradientCheckpointing?: boolean
      unslothOffloadCheckpointing?: boolean
      cpuOffloadCheckpointing?: boolean
      mixedPrecision?: "bf16" | "fp16" | "fp32"
      useCmmd?: boolean
      validationSeed?: number | null
      validationSampleSteps?: number | null
      validationCfgScale?: number | null
      // Upstream-locked / risky fields (B5 cut-locks). Most of these
      // mirror base.toml defaults that anima_lora's argparse can't
      // actually flip off; the editor surfaces them with 🔒 / ⚠️
      // badges so the user knows what's a no-op vs what's risky.
      maskedLoss?: boolean
      torchCompile?: boolean
      skipCacheCheck?: boolean
      dataloaderPinMemory?: boolean
      persistentDataLoaderWorkers?: boolean
      trimCrossattnKv?: boolean
      saveModelAs?: "safetensors"
      savePrecision?: "bf16" | "fp16" | "fp32"
      logEveryNSteps?: number
      keepTokens?: number
      captionExtension?: string
      validationSplitNum?: number
      enableBucket?: boolean
      pathPattern?: string
      // method = lora sub-config (default OrthoLoRA + T-LoRA stack).
      // The ``algorithm`` enum is the authoritative selector; the legacy
      // ``useX`` booleans are kept as optional deprecated shadows for
      // compat with older YAML / API callers but the form drives the
      // dropdown off ``algorithm``.
      lora?: {
        algorithm?:
          | "lora"
          | "ortho"
          | "dora"
          | "ia3"
          | "lokr"
          | "loha"
          | "dylora"
          | "full"
          | "diag_oft"
          | "boft"
          | "glora"
          | "vera"
        useOrtho?: boolean | null
        useDora?: boolean | null
        useIa3?: boolean | null
        useLokr?: boolean | null
        useLoha?: boolean | null
        useDylora?: boolean | null
        useFull?: boolean | null
        useDiagOft?: boolean | null
        useBoft?: boolean | null
        useGlora?: boolean | null
        useVera?: boolean | null
        lokrFactor?: number
        boftFactors?: number
        useTimestepMask?: boolean
        minRank?: number
        alphaRankScale?: number
      }
      postfix?: {
        mode?: "postfix" | "cond"
        condHiddenDim?: number
        splicePosition?: "front_of_padding" | "after_padding"
        orthoBasis?: "svd_te" | "random" | "identity"
        teCacheDir?: string | null
        svdNumFiles?: number
        orthoBasisSeed?: number
        lambdaInit?: number
      }
      chimera?: {
        balanceWContent?: number
        balanceWFreq?: number
        balanceLossWarmupRatio?: number
        feiFeatureDim?: number
        sigmaFeatureDim?: number
      }
      easycontrol?: {
        bCondInit?: number
        condScale?: number
        applyFfnLora?: boolean
        condTokenCount?: number
        dropP?: number
        condNoiseMax?: number
      }
      ipAdapter?: {
        encoder?: "PE-Core-L14-336" | "PE-Core-G14-448"
        resamplerLayers?: number
        resamplerHeads?: number
        ipScale?: number
        imageDropP?: number
        gateLr?: number
        featuresCacheToDisk?: boolean
      }
      // Turbo distillation — when set, compiler routes through
      // scripts/distill_turbo.py instead of train.py.
      turbo?: {
        iterations?: number
        batchSize?: number
        seed?: number
        useCustomDownAutograd?: boolean
        studentRank?: number
        studentAlpha?: number
        fakeRank?: number
        fakeAlpha?: number
        attnMode?: "flash" | "torch" | "flex" | "sageattn" | "xformers"
        studentSteps?: number
        teacherCfg?: number
        tauCaStrategy?: "above_t" | "uniform"
        tauDmStrategy?: "uniform" | "above_t"
        tauCaMinGap?: number
        tauCaSkipAboveT?: number
        studentLr?: number
        fakeLr?: number
        fakeStepsPerStudentStep?: number
        alphaWarmupSteps?: number
        weightDecay?: number
        gradClip?: number
        tDistribution?: "uniform" | "sigmoid"
        sigmoidScale?: number
        saveEvery?: number
        logInterval?: number
      }
    }
  }
  resume?: {
    saveState?: boolean
    saveStateAtEnd?: boolean
    saveStateEveryNEpochs?: number | null
    resumeFrom?: string | null
    saveLastNEpochsState?: number | null
    saveLastNStepsState?: number | null
    skipUntilInitialStep?: boolean
    initialEpoch?: number | null
    initialStep?: number | null
  }
  validation?: {
    everyNEpochs?: number
    everyNSteps?: number | null
    maxSamples?: number | null
    seed?: number | null
  }
  optimization?: {
    torchCompile?: boolean
    fusedBackwardPass?: boolean
    fullBf16?: boolean
    fullFp16?: boolean
    blocksToSwap?: number
    fp8Base?: boolean
    fp8BaseUnet?: boolean
    fp8Scaled?: boolean
    fp8VlTextEncoder?: boolean
    lowram?: boolean
    highvram?: boolean
    noHalfVae?: boolean
    disableMmapLoadSafetensors?: boolean
    cpuOffloadCheckpointing?: boolean
    unslothOffloadCheckpointing?: boolean
    cacheTextEncoderOutputs?: boolean
    cacheTextEncoderOutputsToDisk?: boolean
  }
  attention?: {
    training?: string
    split?: boolean
  }
  dataloader?: {
    numWorkers?: number
    persistentWorkers?: boolean
    vaeBatchSize?: number
    textEncoderBatchSize?: number | null
    cacheShuffleNum?: number
    mapNumProc?: number | null
  }
  augmentation?: {
    flip?: boolean
    color?: boolean
    randomCrop?: boolean
    faceCropAugRange?: string | null
    alphaMask?: boolean
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

// pydantic returns validation error `loc` paths using the Python field name
// (snake_case). Form paths and labels are camelCase, so we normalise the loc
// segments before keying the map. Numeric indices and modelPaths' verbatim
// snake keys must pass through unchanged.
const _MODEL_PATHS_VERBATIM = /^[a-z][a-z0-9]*(_[a-z0-9]+)+$/
function _snakeToCamelSegment(s: string, parent: string | number | undefined): string {
  if (typeof s !== "string") return s
  // Don't mangle keys nested under modelPaths — those are dp TOML literals.
  if (parent === "modelPaths" && _MODEL_PATHS_VERBATIM.test(s)) return s
  if (!s.includes("_")) return s
  return s.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase())
}

export function buildErrorMap(errors: ValidationFieldError[] | undefined): ErrorMap {
  const m = new Map<string, string[]>()
  if (!errors) return m
  for (const e of errors) {
    const segments: (string | number)[] = []
    for (const seg of e.loc) {
      if (typeof seg === "number") {
        segments.push(seg)
      } else {
        segments.push(_snakeToCamelSegment(seg, segments[segments.length - 1]))
      }
    }
    const key = segments.join(".")
    const arr = m.get(key) ?? []
    arr.push(e.msg)
    m.set(key, arr)
  }
  return m
}
