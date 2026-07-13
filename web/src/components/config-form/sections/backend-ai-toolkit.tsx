import { memo, useEffect, useMemo, useState } from "react"
import {
  Boxes,
  Cpu,
  Database,
  Gauge,
  Image,
  Plus,
  Save,
  Trash2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { DatasetSourceSelect } from "../dataset-source-select"
import type {
  ConfigFormValue,
  ErrorMap,
  SamplingPromptValue,
  Setter,
} from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  KeyValueTextArea,
  PathInput,
  ResolutionInput,
  Row,
  Section,
  SeedInput,
  TextInput,
  ToggleSwitch,
} from "../widgets"
import { ModelPathPicker } from "../widgets-model-picker"
import { PromptsDialog } from "./prompts-dialog"

const NETWORK_TYPES = [
  { value: "lora", label: "LoRA" },
  { value: "dora", label: "DoRA" },
  { value: "loha", label: "LoHA" },
  { value: "lokr", label: "LoKr" },
  { value: "lorm", label: "LoRM" },
] as const

const OPTIMIZERS = [
  { value: "adamw8bit", label: "AdamW 8-bit" },
  { value: "adamw", label: "AdamW" },
  { value: "adam", label: "Adam" },
  { value: "adam8bit", label: "Adam 8-bit" },
  { value: "prodigy", label: "Prodigy" },
  { value: "prodigy8bit", label: "Prodigy 8-bit" },
  { value: "adagrad", label: "Adagrad" },
  { value: "adafactor", label: "Adafactor" },
  { value: "automagic", label: "Automagic" },
  { value: "automagic2", label: "Automagic 2" },
  { value: "automagic3", label: "Automagic 3" },
] as const

const STANDARD_BETA_OPTIMIZERS = new Set([
  "adam",
  "adam8bit",
  "adamw",
  "adamw8bit",
  "prodigy",
  "prodigy8bit",
])

const BETA2_OPTIMIZERS = new Set(["automagic", "automagic2", "automagic3"])

const SCHEDULERS = [
  { value: "constant", label: "constant" },
  { value: "constant_with_warmup", label: "constant_with_warmup" },
  { value: "linear", label: "linear" },
  { value: "cosine", label: "cosine" },
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
] as const

const PRECISIONS = [
  { value: "bf16", label: "bf16" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

const MODEL_QTYPES = [
  { value: "qfloat8", label: "qfloat8" },
  { value: "float8", label: "float8" },
  { value: "int8", label: "int8" },
  { value: "uint8", label: "uint8" },
  { value: "uint4", label: "uint4" },
] as const

const TEXT_ENCODER_QTYPES = [
  { value: "qfloat8", label: "qfloat8" },
  { value: "qint8", label: "qint8" },
  { value: "qint4", label: "qint4" },
] as const

const CONTENT_MODES = [
  { value: "balanced", label: "均衡" },
  { value: "style", label: "风格" },
  { value: "content", label: "内容 / 角色" },
] as const

const TIMESTEP_TYPES = [
  { value: "sigmoid", label: "sigmoid" },
  { value: "linear", label: "linear" },
  { value: "lognorm_blend", label: "lognorm_blend" },
  { value: "weighted", label: "weighted" },
  { value: "next_sample", label: "next_sample" },
  { value: "one_step", label: "one_step" },
  { value: "two_step", label: "two_step" },
  { value: "four_step", label: "four_step" },
  { value: "eight_step", label: "eight_step" },
] as const

const LOSS_TYPES = [
  { value: "mse", label: "MSE" },
  { value: "mae", label: "MAE" },
  { value: "pseudo_huber", label: "Pseudo-Huber" },
  { value: "wavelet", label: "Wavelet" },
  { value: "mean_flow", label: "Mean Flow" },
] as const

const IMAGE_FORMATS = [
  { value: "jpg", label: "JPG" },
  { value: "png", label: "PNG" },
  { value: "webp", label: "WebP" },
] as const

const SAVE_DTYPES = [
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "float", label: "float32" },
] as const

interface Props {
  value: ConfigFormValue
  set: Setter
  errorMap: ErrorMap
}

export const BackendAiToolkitFields = memo(function BackendAiToolkitFields({
  value,
  set,
  errorMap,
}: Props) {
  return (
    <>
      <AiToolkitModelFields value={value} set={set} errorMap={errorMap} />
      <AiToolkitDatasetFields value={value} set={set} errorMap={errorMap} />
      <AiToolkitNetworkFields value={value} set={set} errorMap={errorMap} />
      <AiToolkitTrainFields value={value} set={set} errorMap={errorMap} />
      <AiToolkitSamplingFields value={value} set={set} errorMap={errorMap} />
      <AiToolkitOutputFields value={value} set={set} errorMap={errorMap} />
    </>
  )
})

function AiToolkitModelFields({ value, set, errorMap }: Props) {
  const model = value.backend?.aiToolkit?.model ?? {}
  return (
    <Section icon={<Cpu className="size-3.5" />} title="Krea2 模型" subtitle="检查点、组件与显存策略">
      <Row label="基础模型" required errors={errorMap.get("baseModel.checkpoint")}>
        <ModelPathPicker
          value={value.baseModel?.checkpoint ?? ""}
          onChange={(next) => set(["baseModel", "checkpoint"], next)}
          placeholder="krea/Krea-2-Raw"
        />
      </Row>
      <Row label="Assistant LoRA" description="可选。本地文件或 owner/repo/file 格式。">
        <PathInput
          value={model.assistantLoraPath ?? ""}
          onChange={(next) => set(["backend", "aiToolkit", "model", "assistantLoraPath"], next || null)}
          placeholder="可选"
        />
      </Row>
      <Row label="量化 DiT">
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={model.quantize ?? true}
            onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "quantize"], next)}
          />
          {(model.quantize ?? true) && (
            <EnumSelect
              value={model.qtype ?? "qfloat8"}
              onChange={(next) => set(["backend", "aiToolkit", "model", "qtype"], next)}
              options={MODEL_QTYPES}
            />
          )}
        </div>
      </Row>
      <Row label="量化文本编码器">
        <div className="flex items-center gap-3">
          <ToggleSwitch
            checked={model.quantizeTextEncoder ?? true}
            onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "quantizeTextEncoder"], next)}
          />
          {(model.quantizeTextEncoder ?? true) && (
            <EnumSelect
              value={model.qtypeTextEncoder ?? "qfloat8"}
              onChange={(next) => set(["backend", "aiToolkit", "model", "qtypeTextEncoder"], next)}
              options={TEXT_ENCODER_QTYPES}
            />
          )}
        </div>
      </Row>
      <Row label="低显存模式" description="按阶段把未使用模块移到 CPU。">
        <ToggleSwitch
          checked={model.lowVram ?? false}
          onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "lowVram"], next)}
        />
      </Row>
      <Row label="层卸载" description="按比例卸载 DiT 与文本编码器层。">
        <ToggleSwitch
          checked={model.layerOffloading ?? false}
          onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "layerOffloading"], next)}
        />
      </Row>
      {model.layerOffloading && (
        <div className="grid gap-3 sm:grid-cols-2">
          <Row label="DiT 卸载比例">
            <FloatInput min={0} max={1} step={0.05} value={model.layerOffloadingTransformerPercent ?? 1} onChange={(next) => set(["backend", "aiToolkit", "model", "layerOffloadingTransformerPercent"], next ?? 1)} />
          </Row>
          <Row label="文本编码器卸载比例">
            <FloatInput min={0} max={1} step={0.05} value={model.layerOffloadingTextEncoderPercent ?? 1} onChange={(next) => set(["backend", "aiToolkit", "model", "layerOffloadingTextEncoderPercent"], next ?? 1)} />
          </Row>
        </div>
      )}
      <AiDetails title="组件路径与编译">
        <Row label="检查点文件名" description="仓库内权重文件不是默认名称时填写。">
          <TextInput value={model.checkpointFilename ?? ""} onChange={(next) => set(["backend", "aiToolkit", "model", "checkpointFilename"], next || null)} placeholder="自动识别" />
        </Row>
        <Row label="VAE 仓库或目录">
          <PathInput value={model.vaePath ?? ""} onChange={(next) => set(["backend", "aiToolkit", "model", "vaePath"], next || null)} placeholder="Qwen/Qwen-Image" />
        </Row>
        <Row label="文本编码器仓库或目录">
          <PathInput value={model.textEncoderPath ?? ""} onChange={(next) => set(["backend", "aiToolkit", "model", "textEncoderPath"], next || null)} placeholder="使用后端默认值" />
        </Row>
        <Row label="最大文本长度">
          <IntInput min={1} max={4096} value={model.maxTextLength ?? 512} onChange={(next) => set(["backend", "aiToolkit", "model", "maxTextLength"], next ?? 512)} />
        </Row>
        <Row label="torch.compile">
          <ToggleSwitch checked={model.compile ?? value.optimization?.torchCompile ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "compile"], next)} />
        </Row>
        {(model.compile ?? value.optimization?.torchCompile ?? false) && (
          <>
            <Row label="按 Block 编译">
              <ToggleSwitch checked={model.blockCompile ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "blockCompile"], next)} />
            </Row>
            <Row label="编译模式">
              <EnumSelect value={model.compileMode ?? "default"} onChange={(next) => set(["backend", "aiToolkit", "model", "compileMode"], next)} options={[
                { value: "default", label: "default" },
                { value: "reduce-overhead", label: "reduce-overhead" },
                { value: "max-autotune", label: "max-autotune" },
              ]} />
            </Row>
            <Row label="动态 Shape">
              <ToggleSwitch checked={model.compileDynamic ?? true} onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "compileDynamic"], next)} />
            </Row>
            <Row label="Full graph">
              <ToggleSwitch checked={model.compileFullgraph ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "model", "compileFullgraph"], next)} />
            </Row>
            <Row label="Dynamo 缓存上限">
              <IntInput min={1} value={model.cacheSizeLimit ?? null} onChange={(next) => set(["backend", "aiToolkit", "model", "cacheSizeLimit"], next)} placeholder="默认" />
            </Row>
          </>
        )}
      </AiDetails>
    </Section>
  )
}

function AiToolkitDatasetFields({ value, set, errorMap }: Props) {
  const dataset = value.dataset ?? {}
  const options = value.backend?.aiToolkit?.dataset ?? {}
  const subsets = dataset.subsets ?? []
  const legacyResolution = useMemo(
    () => toAiResolution(dataset.resolution ?? [1024, 1024]),
    [dataset.resolution],
  )
  return (
    <Section icon={<Database className="size-3.5" />} title="ai-toolkit 数据集" subtitle="分桶、标注与缓存">
      <Row
        label="主数据集"
        required={subsets.length === 0}
        description={
          subsets.length > 0
            ? "当前训练使用下方多数据集；此路径不会参与本次训练。"
            : undefined
        }
        errors={errorMap.get("dataset.source")}
      >
        <DatasetSourceSelect value={dataset.source} onChange={(next) => set(["dataset", "source"], next)} />
      </Row>
      <Row label="目标分辨率" description="单个值或逗号分隔列表；表示分桶像素预算，不是宽×高。">
        <ResolutionListInput
          value={options.resolutions ?? [legacyResolution]}
          onChange={(next) => set(["backend", "aiToolkit", "dataset", "resolutions"], next)}
        />
      </Row>
      <Row label="重复次数">
        <IntInput min={1} value={dataset.numRepeats ?? 1} onChange={(next) => set(["dataset", "numRepeats"], next ?? 1)} />
      </Row>
      <Row label="标注扩展名">
        <TextInput value={dataset.caption?.ext ?? ".txt"} onChange={(next) => set(["dataset", "caption", "ext"], next)} className="w-32" />
      </Row>
      <Row label="整条标注丢弃率" description="随机将整条 caption 置空；不是单个标签丢弃。">
        <FloatInput min={0} max={1} step={0.05} value={dataset.caption?.dropRate ?? 0} onChange={(next) => set(["dataset", "caption", "dropRate"], next ?? 0)} />
      </Row>
      <Row label="触发词">
        <TextInput value={options.triggerWord ?? ""} onChange={(next) => set(["backend", "aiToolkit", "dataset", "triggerWord"], next || null)} placeholder="可选" />
      </Row>
      <AiDetails title="标注、增强与缓存">
        <Row
          label="清理标签"
          description="每行一个。训练前生成过滤后的 caption 镜像，源标注文件不变。"
          errors={errorMap.get("dataset.caption.dropTokens")}
        >
          <Textarea
            className="max-w-2xl font-mono"
            rows={4}
            placeholder={"1girl\nlooking at viewer\n2d, anime style"}
            value={(dataset.caption?.dropTokens ?? []).join("\n")}
            onChange={(event) => {
              const dropTokens = event.target.value
                .split("\n")
                .map((line) => line.trim())
                .filter(Boolean)
              set(["dataset", "caption", "dropTokens"], dropTokens)
            }}
          />
        </Row>
        <Row label="默认标注" description="图片缺少标注文件时使用。">
          <TextInput value={options.defaultCaption ?? ""} onChange={(next) => set(["backend", "aiToolkit", "dataset", "defaultCaption"], next || null)} placeholder="可选" />
        </Row>
        <Row label="打乱标签">
          <ToggleSwitch checked={options.shuffleTokens ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "shuffleTokens"], next)} />
        </Row>
        <Row label="单标签丢弃率">
          <FloatInput min={0} max={1} step={0.05} value={options.tokenDropoutRate ?? 0} onChange={(next) => set(["backend", "aiToolkit", "dataset", "tokenDropoutRate"], next ?? 0)} />
        </Row>
        <Row label="保留前 N 个标签">
          <IntInput min={0} value={options.keepTokens ?? 0} onChange={(next) => set(["backend", "aiToolkit", "dataset", "keepTokens"], next ?? 0)} />
        </Row>
        <Row label="启用分桶">
          <ToggleSwitch checked={options.buckets ?? true} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "buckets"], next)} />
        </Row>
        <Row label="随机裁剪">
          <ToggleSwitch checked={options.randomCrop ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "randomCrop"], next)} />
        </Row>
        <Row label="随机缩放">
          <ToggleSwitch checked={options.randomScale ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "randomScale"], next)} />
        </Row>
        <Row label="图像缩放倍率">
          <FloatInput min={0.01} step={0.05} value={options.scale ?? 1} onChange={(next) => set(["backend", "aiToolkit", "dataset", "scale"], next ?? 1)} />
        </Row>
        <Row label="水平翻转">
          <ToggleSwitch checked={options.flipX ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "flipX"], next)} />
        </Row>
        <Row label="垂直翻转">
          <ToggleSwitch checked={options.flipY ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "flipY"], next)} />
        </Row>
        <Row label="潜变量缓存到内存">
          <ToggleSwitch checked={options.cacheLatents ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "cacheLatents"], next)} />
        </Row>
        <Row label="潜变量缓存到磁盘">
          <ToggleSwitch checked={value.cacheLatentsToDisk ?? false} onCheckedChange={(next) => set(["cacheLatentsToDisk"], next)} />
        </Row>
        {(options.cacheLatents || value.cacheLatentsToDisk) && (
          <Row label="缓存时保留原图" description="需要图像增强或掩码时启用。">
            <ToggleSwitch checked={options.loadImageWhenCachingLatents ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "loadImageWhenCachingLatents"], next)} />
          </Row>
        )}
        <Row label="缓存文本编码">
          <ToggleSwitch checked={options.cacheTextEmbeddings ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "dataset", "cacheTextEmbeddings"], next)} />
        </Row>
        <Row label="数据加载进程">
          <IntInput
            min={0}
            value={options.numWorkers ?? 2}
            onChange={(next) => {
              const numWorkers = next ?? 2
              set(["backend", "aiToolkit", "dataset"], {
                ...options,
                numWorkers,
                prefetchFactor:
                  numWorkers === 0 ? null : (options.prefetchFactor ?? 2),
              })
            }}
          />
        </Row>
        {(options.numWorkers ?? 2) > 0 && (
          <Row label="预取批数">
            <IntInput min={1} value={options.prefetchFactor ?? 2} onChange={(next) => set(["backend", "aiToolkit", "dataset", "prefetchFactor"], next ?? 2)} />
          </Row>
        )}
      </AiDetails>
      <Row label="正则化数据集" description="可选。作为 is_reg 数据集加入训练。">
        <PathInput value={dataset.regSource ?? ""} onChange={(next) => set(["dataset", "regSource"], next || null)} placeholder="可选" />
      </Row>
      <AiDetails title={`多数据集 · ${subsets.length}`}>
        <div className="space-y-2">
          {subsets.map((subset, index) => (
            <div key={index} className="rounded-[6px] border border-border/50 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">数据集 #{index + 1}</span>
                <Button type="button" variant="ghost" size="icon" className="size-7" onClick={() => set(["dataset", "subsets"], subsets.filter((_, itemIndex) => itemIndex !== index))} title="删除">
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
              <Row label="路径">
                <PathInput value={subset.path ?? ""} onChange={(next) => set(["dataset", "subsets", index, "path"], next)} placeholder="数据集路径" />
              </Row>
              <Row label="重复次数">
                <IntInput min={1} value={subset.numRepeats ?? 1} onChange={(next) => set(["dataset", "subsets", index, "numRepeats"], next ?? 1)} />
              </Row>
              <Row label="掩码目录">
                <PathInput value={subset.maskPath ?? ""} onChange={(next) => set(["dataset", "subsets", index, "maskPath"], next || null)} placeholder="掩码目录（可选）" />
              </Row>
            </div>
          ))}
          <Button type="button" variant="outline" size="sm" onClick={() => set(["dataset", "subsets"], [...subsets, { path: "", numRepeats: 1 }])}>
            <Plus className="size-3.5" /> 添加数据集
          </Button>
          {subsets.length > 0 && <p className="text-[11px] text-muted-foreground">存在列表时，训练使用列表中的数据集，不再使用主数据集。</p>}
        </div>
      </AiDetails>
    </Section>
  )
}

function AiToolkitNetworkFields({ value, set, errorMap }: Props) {
  const network = value.network ?? {}
  const options = value.backend?.aiToolkit?.network ?? {}
  const type = network.type ?? "lora"
  return (
    <Section icon={<Boxes className="size-3.5" />} title="训练网络" subtitle="LoRA、LyCORIS 与续训">
      <Row label="网络类型">
        <EnumSelect value={type} onChange={(next) => set(["network", "type"], next)} options={NETWORK_TYPES} />
      </Row>
      <Row label="Rank" errors={errorMap.get("network.rank")}>
        <IntInput min={1} max={512} value={network.rank ?? 16} onChange={(next) => set(["network", "rank"], next ?? 16)} />
      </Row>
      <Row label="Alpha" errors={errorMap.get("network.alpha")}>
        <IntInput min={1} value={network.alpha ?? 16} onChange={(next) => set(["network", "alpha"], next ?? 16)} />
      </Row>
      <Row label="Network Dropout">
        <FloatInput min={0} max={0.99} step={0.05} value={network.networkDropout ?? 0} onChange={(next) => set(["network", "networkDropout"], next ?? 0)} />
      </Row>
      <Row label="Rank Dropout">
        <FloatInput min={0} max={0.99} step={0.05} value={network.rankDropout ?? 0} onChange={(next) => set(["network", "rankDropout"], next ?? 0)} />
      </Row>
      <Row label="Module Dropout">
        <FloatInput min={0} max={0.99} step={0.05} value={network.moduleDropout ?? 0} onChange={(next) => set(["network", "moduleDropout"], next ?? 0)} />
      </Row>
      <Row label="预训练 LoRA" description="从已有权重继续训练。">
        <PathInput value={network.initFrom ?? ""} onChange={(next) => set(["network", "initFrom"], next || null)} placeholder="可选" />
      </Row>
      {type === "lokr" && (
        <AiDetails title="LoKr 参数" open>
          <Row label="分解因子" description="-1 自动选择最大因子。">
            <IntInput min={-1} value={options.lokrFactor ?? -1} onChange={(next) => set(["backend", "aiToolkit", "network", "lokrFactor"], next ?? -1)} />
          </Row>
          <Row label="Full rank">
            <ToggleSwitch checked={options.lokrFullRank ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "network", "lokrFullRank"], next)} />
          </Row>
          <Row label="旧版 LoKr 格式">
            <ToggleSwitch checked={options.oldLokrFormat ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "network", "oldLokrFormat"], next)} />
          </Row>
        </AiDetails>
      )}
      {type === "lorm" && (
        <AiDetails title="LoRM 参数" open>
          <Row label="提取模式">
            <EnumSelect value={options.lormExtractMode ?? "fixed"} onChange={(next) => set(["backend", "aiToolkit", "network", "lormExtractMode"], next)} options={[
              { value: "ratio", label: "ratio" },
              { value: "fixed", label: "fixed" },
            ]} />
          </Row>
          <Row label="提取参数">
            <FloatInput min={0.0001} step={0.05} value={options.lormExtractModeParam ?? (options.lormExtractMode === "ratio" ? 0.25 : network.rank ?? 16)} onChange={(next) => set(["backend", "aiToolkit", "network", "lormExtractModeParam"], next)} />
          </Row>
          <Row label="参数量阈值">
            <IntInput min={0} value={options.lormParameterThreshold ?? 0} onChange={(next) => set(["backend", "aiToolkit", "network", "lormParameterThreshold"], next ?? 0)} />
          </Row>
        </AiDetails>
      )}
    </Section>
  )
}

function AiToolkitTrainFields({ value, set, errorMap }: Props) {
  const schedule = value.schedule ?? {}
  const optimizer = value.optimizer ?? {}
  const train = value.backend?.aiToolkit?.train ?? {}
  const [beta1 = 0.9, beta2 = 0.999] = optimizer.betas ?? []
  const epochs = schedule.epochs ?? 10
  const lrScheduler = train.lrScheduler ?? optimizer.schedule ?? "constant"
  return (
    <Section icon={<Gauge className="size-3.5" />} title="训练参数" subtitle="步数、优化器与损失策略">
      <Row label="训练回合" description="启动后按实际数据集、重复、批大小与梯度累积计算总步数。" errors={errorMap.get("schedule.epochs")}>
        <IntInput min={1} value={epochs} onChange={(next) => set(["schedule", "epochs"], next ?? epochs)} />
      </Row>
      <Row label="最大训练步数" description="可选上限；达到上限会提前结束，不会覆盖训练回合。" errors={errorMap.get("schedule.maxSteps")}>
        <IntInput min={1} value={schedule.maxSteps ?? null} onChange={(next) => set(["schedule", "maxSteps"], next)} placeholder="不限制" />
      </Row>
      <Row label="批大小">
        <IntInput min={1} value={schedule.batchSize ?? 1} onChange={(next) => set(["schedule", "batchSize"], next ?? 1)} />
      </Row>
      <Row label="梯度累积">
        <IntInput min={1} value={schedule.gradAccum ?? 1} onChange={(next) => set(["schedule", "gradAccum"], next ?? 1)} />
      </Row>
      <Row label="随机种子">
        <IntInput value={schedule.seed ?? null} onChange={(next) => set(["schedule", "seed"], next)} placeholder="随机" />
      </Row>
      <Row label="训练精度">
        <EnumSelect value={value.precision ?? "bf16"} onChange={(next) => set(["precision"], next)} options={PRECISIONS} />
      </Row>
      <Row label="梯度检查点">
        <ToggleSwitch checked={value.gradientCheckpointing ?? true} onCheckedChange={(next) => set(["gradientCheckpointing"], next)} />
      </Row>
      <Row label="优化器">
        <EnumSelect value={optimizer.type ?? "adamw8bit"} onChange={(next) => set(["optimizer", "type"], next)} options={OPTIMIZERS} />
      </Row>
      <Row label="学习率" errors={errorMap.get("optimizer.lr.unet")}>
        <FloatInput min={0} step={1e-5} value={optimizer.lr?.unet ?? 1e-4} onChange={(next) => set(["optimizer", "lr", "unet"], next ?? 1e-4)} />
      </Row>
      <Row label="学习率调度">
        <EnumSelect value={lrScheduler} onChange={(next) => set(["backend", "aiToolkit", "train", "lrScheduler"], next)} options={SCHEDULERS} />
      </Row>
      {lrScheduler === "constant_with_warmup" && (
        <Row label="预热步数">
          <IntInput min={0} value={optimizer.warmupSteps ?? 100} onChange={(next) => set(["optimizer", "warmupSteps"], next ?? 0)} />
        </Row>
      )}
      <Row label="训练侧重">
        <EnumSelect value={train.contentOrStyle ?? "balanced"} onChange={(next) => set(["backend", "aiToolkit", "train", "contentOrStyle"], next)} options={CONTENT_MODES} />
      </Row>
      <Row label="时间步分布">
        <EnumSelect value={train.timestepType ?? "sigmoid"} onChange={(next) => set(["backend", "aiToolkit", "train", "timestepType"], next)} options={TIMESTEP_TYPES} />
      </Row>
      <Row label="损失类型">
        <EnumSelect value={train.lossType ?? "mse"} onChange={(next) => set(["backend", "aiToolkit", "train", "lossType"], next)} options={LOSS_TYPES} />
      </Row>
      <AiDetails title="优化器与稳定性">
        <Row label="最大梯度范数">
          <FloatInput min={0} step={0.1} value={optimizer.maxGradNorm ?? 1} onChange={(next) => set(["optimizer", "maxGradNorm"], next ?? 1)} />
        </Row>
        <Row label="Weight decay">
          <FloatInput min={0} step={0.001} value={optimizer.weightDecay ?? 0} onChange={(next) => set(["optimizer", "weightDecay"], next ?? 0)} />
        </Row>
        {STANDARD_BETA_OPTIMIZERS.has(optimizer.type ?? "adamw8bit") && (
          <Row label="Betas">
            <div className="flex gap-2">
              <FloatInput min={0} max={1} step={0.001} value={beta1} onChange={(next) => set(["optimizer", "betas"], [next ?? 0.9, beta2])} />
              <FloatInput min={0} max={1} step={0.001} value={beta2} onChange={(next) => set(["optimizer", "betas"], [beta1, next ?? 0.999])} />
            </div>
          </Row>
        )}
        {BETA2_OPTIMIZERS.has(optimizer.type ?? "") && (
          <Row label="Beta2">
            <FloatInput min={0} max={1} step={0.001} value={beta2} onChange={(next) => set(["optimizer", "betas"], [beta1, next ?? 0.999])} />
          </Row>
        )}
        <Row label="优化器参数">
          <KeyValueTextArea value={optimizer.optimizerArgs} onChange={(next) => set(["optimizer", "optimizerArgs"], next)} placeholder={"decouple = true\nweight_decay = 0.01"} />
        </Row>
        <Row label="调度器参数">
          <KeyValueTextArea value={optimizer.schedulerArgs} onChange={(next) => set(["optimizer", "schedulerArgs"], next)} placeholder={"factor = 1.0"} />
        </Row>
        {lrScheduler === "cosine_with_restarts" && (
          <Row label="重启周期数">
            <IntInput min={1} value={optimizer.schedulerNumCycles ?? 1} onChange={(next) => set(["optimizer", "schedulerNumCycles"], next ?? 1)} />
          </Row>
        )}
        {(lrScheduler === "cosine" || lrScheduler === "cosine_with_restarts") && (
          <Row label="最小学习率比例">
            <FloatInput min={0} max={1} step={0.01} value={optimizer.schedulerMinLrRatio ?? null} onChange={(next) => set(["optimizer", "schedulerMinLrRatio"], next)} placeholder="默认" />
          </Row>
        )}
        <Row label="调度窗口步数">
          <IntInput min={1} value={schedule.lrDecaySteps ?? null} onChange={(next) => set(["schedule", "lrDecaySteps"], next)} placeholder="使用总步数" />
        </Row>
        <Row label="最小去噪步">
          <IntInput min={0} value={train.minDenoisingSteps ?? 0} onChange={(next) => set(["backend", "aiToolkit", "train", "minDenoisingSteps"], next ?? 0)} />
        </Row>
        <Row label="最大去噪步">
          <IntInput min={0} value={train.maxDenoisingSteps ?? 999} onChange={(next) => set(["backend", "aiToolkit", "train", "maxDenoisingSteps"], next ?? 999)} />
        </Row>
        <Row label="Min-SNR Gamma">
          <FloatInput min={0.0001} step={0.1} value={train.minSnrGamma ?? null} onChange={(next) => set(["backend", "aiToolkit", "train", "minSnrGamma"], next)} placeholder="关闭" />
        </Row>
        <Row label="Noise offset">
          <FloatInput min={0} step={0.01} value={train.noiseOffset ?? 0} onChange={(next) => set(["backend", "aiToolkit", "train", "noiseOffset"], next ?? 0)} />
        </Row>
        <Row label="Prompt dropout">
          <FloatInput min={0} max={1} step={0.05} value={train.promptDropoutProb ?? 0} onChange={(next) => set(["backend", "aiToolkit", "train", "promptDropoutProb"], next ?? 0)} />
        </Row>
        <Row label="训练后卸载文本编码器">
          <ToggleSwitch checked={train.unloadTextEncoder ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "train", "unloadTextEncoder"], next)} />
        </Row>
        <Row label="EMA">
          <ToggleSwitch checked={train.useEma ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "train", "useEma"], next)} />
        </Row>
        {train.useEma && (
          <>
            <Row label="EMA decay">
              <FloatInput min={0.0001} max={0.999999} step={0.0001} value={train.emaDecay ?? 0.999} onChange={(next) => set(["backend", "aiToolkit", "train", "emaDecay"], next ?? 0.999)} />
            </Row>
            <Row label="EMA feedback">
              <ToggleSwitch checked={train.emaUseFeedback ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "train", "emaUseFeedback"], next)} />
            </Row>
            <Row label="EMA 参数倍率">
              <FloatInput min={0.0001} step={0.01} value={train.emaParamMultiplier ?? 1} onChange={(next) => set(["backend", "aiToolkit", "train", "emaParamMultiplier"], next ?? 1)} />
            </Row>
          </>
        )}
        <Row label="损失裁剪上限">
          <FloatInput min={0.0001} step={0.1} value={train.maxLoss ?? null} onChange={(next) => set(["backend", "aiToolkit", "train", "maxLoss"], next)} placeholder="关闭" />
        </Row>
      </AiDetails>
    </Section>
  )
}

function AiToolkitSamplingFields({ value, set, errorMap }: Props) {
  const sampling = value.sampling ?? {}
  const sampleOptions = value.backend?.aiToolkit?.sample ?? {}
  const train = value.backend?.aiToolkit?.train ?? {}
  const prompts = sampling.prompts ?? []
  const [promptsOpen, setPromptsOpen] = useState(false)
  const resolution: [number, number] = Array.isArray(sampling.resolution) && sampling.resolution.length >= 2
    ? [sampling.resolution[0]!, sampling.resolution[1]!]
    : [1024, 1024]
  return (
    <Section icon={<Image className="size-3.5" />} title="采样预览" subtitle="按回合或步数生成 Krea2 预览">
      <Row label="启用采样">
        <ToggleSwitch checked={sampling.enabled ?? true} onCheckedChange={(next) => set(["sampling", "enabled"], next)} />
      </Row>
      {sampling.enabled !== false && (
        <>
          <Row label="每 N 回合采样" description="可选；与按步采样并行。留空可只按步采样。" errors={errorMap.get("sampling.everyNEpochs")}>
            <IntInput min={1} value={sampling.everyNEpochs ?? null} onChange={(next) => set(["sampling", "everyNEpochs"], next)} placeholder="关闭" />
          </Row>
          <Row label="每 N 步采样" description="可选；留空则不按步采样。">
            <IntInput min={1} value={sampling.everyNSteps ?? null} onChange={(next) => set(["sampling", "everyNSteps"], next)} placeholder="关闭" />
          </Row>
          <Row label="训练前基线采样">
            <ToggleSwitch checked={sampling.atFirst ?? false} onCheckedChange={(next) => set(["sampling", "atFirst"], next)} />
          </Row>
          <Row label="提示词">
            <Button type="button" variant="outline" size="sm" onClick={() => setPromptsOpen(true)}>
              编辑 · {prompts.length}
            </Button>
          </Row>
          <Row label="分辨率">
            <ResolutionInput value={resolution} onChange={(next) => set(["sampling", "resolution"], next)} />
          </Row>
          <Row label="采样步数">
            <IntInput min={1} value={sampling.inferenceSteps ?? 28} onChange={(next) => set(["sampling", "inferenceSteps"], next ?? 28)} />
          </Row>
          <Row label="CFG">
            <FloatInput min={0} step={0.1} value={sampling.inferenceCfg ?? 4.5} onChange={(next) => set(["sampling", "inferenceCfg"], next ?? 4.5)} />
          </Row>
          <Row label="随机种子">
            <SeedInput value={sampling.seed ?? -1} onChange={(next) => set(["sampling", "seed"], next)} />
          </Row>
          <Row label="图片格式">
            <EnumSelect value={sampleOptions.format ?? "jpg"} onChange={(next) => set(["backend", "aiToolkit", "sample", "format"], next)} options={IMAGE_FORMATS} />
          </Row>
          <Row label="提示词递增种子">
            <ToggleSwitch checked={sampleOptions.walkSeed ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "sample", "walkSeed"], next)} />
          </Row>
          <Row label="预览 LoRA 权重">
            <FloatInput min={0} step={0.1} value={sampleOptions.networkMultiplier ?? 1} onChange={(next) => set(["backend", "aiToolkit", "sample", "networkMultiplier"], next ?? 1)} />
          </Row>
          <Row label="恢复训练时强制基线采样">
            <ToggleSwitch checked={train.forceFirstSample ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "train", "forceFirstSample"], next)} />
          </Row>
        </>
      )}
      <PromptsDialog
        open={promptsOpen}
        onOpenChange={setPromptsOpen}
        initial={prompts}
        defaultResolution={resolution}
        triggerWord={value.backend?.aiToolkit?.dataset?.triggerWord ?? null}
        showAdvanced={false}
        onSave={(next: SamplingPromptValue[]) => set(["sampling", "prompts"], next)}
      />
    </Section>
  )
}

function AiToolkitOutputFields({ value, set, errorMap }: Props) {
  const output = value.output ?? {}
  const save = value.backend?.aiToolkit?.save ?? {}
  const logging = value.backend?.aiToolkit?.logging ?? {}
  return (
    <Section icon={<Save className="size-3.5" />} title="保存与日志" subtitle="检查点、Hub 与训练日志">
      <Row label="输出名称">
        <Input value={output.name ?? ""} onChange={(event) => set(["output", "name"], event.target.value)} className="font-mono max-w-md" />
      </Row>
      <Row label="每 N 回合保存">
        <IntInput min={1} value={output.saveEveryNEpochs ?? 1} onChange={(next) => set(["output", "saveEveryNEpochs"], next ?? 1)} />
      </Row>
      <Row label="每 N 步保存" description="可选；可与回合保存同时使用。">
        <IntInput min={1} value={output.saveEveryNSteps ?? null} onChange={(next) => set(["output", "saveEveryNSteps"], next)} placeholder="关闭" />
      </Row>
      <Row label="保留最近 N 个检查点" description="更早的阶段检查点会清理；最终模型始终单独保留。">
        <IntInput min={1} value={output.saveLastNSteps ?? 4} onChange={(next) => set(["output", "saveLastNSteps"], next)} />
      </Row>
      <Row label="保存精度">
        <EnumSelect value={output.saveDtype ?? "fp16"} onChange={(next) => set(["output", "saveDtype"], next)} options={SAVE_DTYPES} />
      </Row>
      <AiDetails title="Hugging Face 与日志">
        <Row label="推送到 Hugging Face">
          <ToggleSwitch checked={save.pushToHub ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "save", "pushToHub"], next)} />
        </Row>
        {save.pushToHub && (
          <>
            <Row label="仓库 ID" errors={errorMap.get("backend.aiToolkit.save.hfRepoId")}>
              <TextInput value={save.hfRepoId ?? ""} onChange={(next) => set(["backend", "aiToolkit", "save", "hfRepoId"], next || null)} placeholder="owner/repository" />
            </Row>
            <Row label="私有仓库">
              <ToggleSwitch checked={save.hfPrivate ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "save", "hfPrivate"], next)} />
            </Row>
          </>
        )}
        <Row label="日志间隔">
          <IntInput min={1} value={logging.logEvery ?? 1} onChange={(next) => set(["backend", "aiToolkit", "logging", "logEvery"], next ?? 1)} />
        </Row>
        <Row label="详细日志">
          <ToggleSwitch checked={logging.verbose ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "logging", "verbose"], next)} />
        </Row>
        <Row label="Weights & Biases">
          <ToggleSwitch checked={logging.useWandb ?? false} onCheckedChange={(next) => set(["backend", "aiToolkit", "logging", "useWandb"], next)} />
        </Row>
        {logging.useWandb && (
          <>
            <Row label="项目名">
              <TextInput value={logging.projectName ?? "lorahub"} onChange={(next) => set(["backend", "aiToolkit", "logging", "projectName"], next || "lorahub")} />
            </Row>
            <Row label="运行名">
              <TextInput value={logging.runName ?? ""} onChange={(next) => set(["backend", "aiToolkit", "logging", "runName"], next || null)} placeholder="可选" />
            </Row>
          </>
        )}
      </AiDetails>
    </Section>
  )
}

function AiDetails({
  title,
  open = false,
  children,
}: {
  title: string
  open?: boolean
  children: React.ReactNode
}) {
  return (
    <details open={open} className="group rounded-[6px] border border-border/50 px-3 py-2">
      <summary className="cursor-pointer select-none text-xs font-medium text-muted-foreground">
        {title}
      </summary>
      <div className="mt-3 space-y-3">{children}</div>
    </details>
  )
}

function ResolutionListInput({
  value,
  onChange,
}: {
  value: number[]
  onChange: (next: number[] | null) => void
}) {
  const serialized = value.join(", ")
  const [draft, setDraft] = useState(serialized)
  useEffect(() => setDraft(serialized), [serialized])

  function commit() {
    const parsed = draft
      .split(",")
      .map((part) => Number.parseInt(part.trim(), 10))
      .filter((item) => Number.isFinite(item) && item >= 64)
    const next = [...new Set(parsed)]
    onChange(next.length > 0 ? next : null)
  }

  return (
    <Input
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault()
          commit()
        }
      }}
      className="font-mono max-w-md"
      placeholder="1024 或 512, 768, 1024"
    />
  )
}

function toAiResolution(value: number[]): number {
  if (value.length < 2) return value[0] ?? 1024
  const [width = 1024, height = 1024] = value
  if (width === height) return width
  return Math.max(64, Math.round(Math.sqrt(width * height) / 16) * 16)
}
