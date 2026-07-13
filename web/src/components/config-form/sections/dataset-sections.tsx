import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  MAX_TOKEN_LENGTH_OPTIONS,
  RESIZE_INTERPOLATION_OPTIONS,
} from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"

type DatasetValue = ConfigFormValue["dataset"]

export function BucketSection({
  bucket,
  set,
  errorMap,
  backendType,
}: {
  bucket: NonNullable<DatasetValue["bucket"]>
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const isKohya = backendType === "kohya"
  const isDiffusionPipe = backendType === "diffusion-pipe"
  const skipImageResolution = bucket.skipImageResolution
  const skipWidth = Array.isArray(skipImageResolution)
    ? skipImageResolution[0]
    : (skipImageResolution ?? null)
  const skipHeight = Array.isArray(skipImageResolution)
    ? skipImageResolution[1]
    : null
  return (
    <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          分桶 Bucket
        </span>
        <ToggleSwitch
          checked={bucket.enabled ?? true}
          onCheckedChange={(value) => set(["dataset", "bucket", "enabled"], value)}
        />
      </div>
      {bucket.enabled !== false && (
        <>
          {!isDiffusionPipe && (
            <div className="grid grid-cols-3 gap-3">
              <div>
                <Label className="text-[11px] text-muted-foreground">最小边</Label>
                <IntInput
                  min={64}
                  value={bucket.min ?? 256}
                  onChange={(value) =>
                    set(["dataset", "bucket", "min"], value ?? 256)
                  }
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">最大边</Label>
                <IntInput
                  min={64}
                  value={bucket.max ?? 2048}
                  onChange={(value) =>
                    set(["dataset", "bucket", "max"], value ?? 2048)
                  }
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">步长</Label>
                <IntInput
                  min={8}
                  value={bucket.step ?? 64}
                  onChange={(value) =>
                    set(["dataset", "bucket", "step"], value ?? 64)
                  }
                />
              </div>
            </div>
          )}
          {isKohya && (
            <>
              <Row label="noUpscale" description="不要把小图放大到桶尺寸。">
                <ToggleSwitch
                  checked={bucket.noUpscale ?? false}
                  onCheckedChange={(value) =>
                    set(["dataset", "bucket", "noUpscale"], value)
                  }
                />
              </Row>
              <Row
                label="跳过图像分辨率"
                description="可选阈值。单值按正方形处理；填写宽高可保留非方形图像。"
              >
                <div className="flex items-center gap-2">
                  <IntInput
                    min={1}
                    value={skipWidth}
                    onChange={(width) => {
                      if (width === null) {
                        set(["dataset", "bucket", "skipImageResolution"], null)
                        return
                      }
                      set(
                        ["dataset", "bucket", "skipImageResolution"],
                        skipHeight === null ? width : [width, skipHeight],
                      )
                    }}
                    placeholder="单边 / 宽"
                    className="w-28"
                  />
                  <span className="text-xs text-muted-foreground">×</span>
                  <IntInput
                    min={1}
                    value={skipHeight}
                    onChange={(height) => {
                      if (height === null) {
                        set(
                          ["dataset", "bucket", "skipImageResolution"],
                          skipWidth,
                        )
                        return
                      }
                      set(
                        ["dataset", "bucket", "skipImageResolution"],
                        [skipWidth ?? height, height],
                      )
                    }}
                    placeholder="高（可选）"
                    className="w-28"
                  />
                </div>
              </Row>
              <Row label="resizeInterpolation" description="PIL 重采样核。">
                <EnumSelect
                  value={bucket.resizeInterpolation ?? ""}
                  onChange={(value) =>
                    set(["dataset", "bucket", "resizeInterpolation"], value || null)
                  }
                  options={RESIZE_INTERPOLATION_OPTIONS}
                />
              </Row>
            </>
          )}
          {isDiffusionPipe && (
            <Row
              label="arBuckets"
              description="显式宽高比列表，逗号分隔；填写后覆盖后端设置中的最小/最大宽高比与桶数。"
              errors={errorMap.get("dataset.bucket.arBuckets")}
            >
              <TextInput
                className="w-64"
                value={(bucket.arBuckets ?? []).join(",")}
                onChange={(raw) => {
                  const list = raw
                    .split(",")
                    .map((part) => part.trim())
                    .filter((part) => part.length > 0)
                    .map((part) => parseFloat(part))
                    .filter((next) => !Number.isNaN(next))
                  set(
                    ["dataset", "bucket", "arBuckets"],
                    list.length ? list : null,
                  )
                }}
                placeholder="（默认）"
              />
            </Row>
          )}
        </>
      )}
    </div>
  )
}

export function CaptionSection({
  caption,
  set,
  errorMap,
  backendType,
}: {
  caption: NonNullable<DatasetValue["caption"]>
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const isKohya = backendType === "kohya"
  const isDiffusionPipe = backendType === "diffusion-pipe"
  const isAnimaLora = backendType === "anima_lora"
  const isAiToolkit = backendType === "ai_toolkit"
  return (
    <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
      <div className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
        标注 Caption
      </div>
      {!isAnimaLora && (
        <>
          <Row label="标注扩展名" description="训练 caption 文件后缀。">
            <Input
              value={caption.ext ?? ".txt"}
              className="font-mono w-32"
              onChange={(event) =>
                set(["dataset", "caption", "ext"], event.target.value)
              }
            />
          </Row>
          {isKohya && (
            <Row label="打乱标签" description="每步随机打乱以逗号分隔的标签。">
              <ToggleSwitch
                checked={caption.shuffle ?? true}
                onCheckedChange={(value) =>
                  set(["dataset", "caption", "shuffle"], value)
                }
              />
            </Row>
          )}
          {!isDiffusionPipe && (
            <Row
              label="丢弃概率"
              description="0-1 之间，每步随机丢弃单个标签的概率。"
              errors={errorMap.get("dataset.caption.dropRate")}
            >
              <FloatInput
                step={0.05}
                value={caption.dropRate ?? 0}
                onChange={(value) =>
                  set(["dataset", "caption", "dropRate"], value ?? 0)
                }
              />
            </Row>
          )}
        </>
      )}

      {(isKohya || isDiffusionPipe || isAiToolkit) && (
      <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
        <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          高级 Caption 选项
        </summary>
        <div className="mt-3 space-y-3.5">
          {isKohya && (
            <>
              <Row
                label="dropoutEveryNEpochs"
                description="每 N 回合做一次 caption dropout。"
                errors={errorMap.get("dataset.caption.dropoutEveryNEpochs")}
              >
                <IntInput
                  min={0}
                  value={caption.dropoutEveryNEpochs ?? 0}
                  onChange={(next) =>
                    set(["dataset", "caption", "dropoutEveryNEpochs"], next ?? 0)
                  }
                />
              </Row>
              <Row
                label="tagDropoutRate"
                description="单个 tag 维度的 dropout。"
                errors={errorMap.get("dataset.caption.tagDropoutRate")}
              >
                <FloatInput
                  step={0.05}
                  value={caption.tagDropoutRate ?? 0}
                  onChange={(next) =>
                    set(["dataset", "caption", "tagDropoutRate"], next ?? 0)
                  }
                />
              </Row>
              <Row
                label="keepTokens"
                description="前 N 个 tag 不打乱。"
                errors={errorMap.get("dataset.caption.keepTokens")}
              >
                <IntInput
                  min={0}
                  value={caption.keepTokens ?? 0}
                  onChange={(next) =>
                    set(["dataset", "caption", "keepTokens"], next ?? 0)
                  }
                />
              </Row>
            </>
          )}
          <Row
            label="dropTokens"
            description="每行一个,完整移除指定字串(大小写不敏感,支持自然语言短语)。例：1girl / looking at viewer。编译时镜像到 captions_sanitized/,源文件不动。"
            errors={errorMap.get("dataset.caption.dropTokens")}
          >
            <textarea
              className="w-full max-w-2xl rounded-[4px] border border-input bg-background px-3 py-2 text-sm font-mono outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground/60"
              rows={4}
              placeholder={"1girl\nlooking at viewer\n2d, anime style"}
              value={(caption.dropTokens ?? []).join("\n")}
              onChange={(event) => {
                const lines = event.target.value
                  .split("\n")
                  .map((line) => line.trim())
                  .filter((line) => line.length > 0)
                set(["dataset", "caption", "dropTokens"], lines)
              }}
            />
          </Row>
          {isKohya && (
            <>
              <Row label="keepTokensSeparator" description="keepTokens 分隔符。">
                <TextInput className="w-32" value={caption.keepTokensSeparator ?? ""} onChange={(value) => set(["dataset", "caption", "keepTokensSeparator"], value || null)} placeholder="," />
              </Row>
              <Row label="secondarySeparator" description="二级分隔符。">
                <TextInput className="w-32" value={caption.secondarySeparator ?? ""} onChange={(value) => set(["dataset", "caption", "secondarySeparator"], value || null)} placeholder="（可选）" />
              </Row>
              <Row label="enableWildcard" description="启用 caption 通配符。">
                <ToggleSwitch checked={caption.enableWildcard ?? false} onCheckedChange={(value) => set(["dataset", "caption", "enableWildcard"], value)} />
              </Row>
              <Row label="prefix" description="所有 caption 之前追加的前缀。">
                <TextInput className="w-64" value={caption.prefix ?? ""} onChange={(value) => set(["dataset", "caption", "prefix"], value || null)} placeholder="（可选）" />
              </Row>
              <Row label="suffix" description="所有 caption 之后追加的后缀。">
                <TextInput className="w-64" value={caption.suffix ?? ""} onChange={(value) => set(["dataset", "caption", "suffix"], value || null)} placeholder="（可选）" />
              </Row>
              <Row label="maxTokenLength" description="默认 75；可选 150 / 225。">
                <EnumSelect
                  value={caption.maxTokenLength === null || caption.maxTokenLength === undefined ? "" : String(caption.maxTokenLength)}
                  onChange={(value) => set(["dataset", "caption", "maxTokenLength"], value ? parseInt(value, 10) : null)}
                  options={MAX_TOKEN_LENGTH_OPTIONS}
                />
              </Row>
              <Row label="tokenWarmupMin" description="token warmup 最小 tag 数。" errors={errorMap.get("dataset.caption.tokenWarmupMin")}>
                <IntInput min={1} value={caption.tokenWarmupMin ?? null} onChange={(next) => set(["dataset", "caption", "tokenWarmupMin"], next)} placeholder="（默认）" />
              </Row>
              <Row label="tokenWarmupStep" description="token warmup 步数。" errors={errorMap.get("dataset.caption.tokenWarmupStep")}>
                <FloatInput step={0.1} value={caption.tokenWarmupStep ?? null} onChange={(next) => set(["dataset", "caption", "tokenWarmupStep"], next)} placeholder="（默认）" />
              </Row>
              <Row label="weighted" description="启用 weighted captions。">
                <ToggleSwitch checked={caption.weighted ?? false} onCheckedChange={(value) => set(["dataset", "caption", "weighted"], value)} />
              </Row>
            </>
          )}
          {isDiffusionPipe && (
            <>
              <Row label="shuffleDelimiter" description="tag shuffle 分隔符。">
                <TextInput className="w-32" value={caption.shuffleDelimiter ?? ""} onChange={(value) => set(["dataset", "caption", "shuffleDelimiter"], value || null)} placeholder="（默认）" />
              </Row>
              <Row label="shuffleTags" description="整 caption 打乱模式。">
                <ToggleSwitch checked={caption.shuffleTags ?? false} onCheckedChange={(value) => set(["dataset", "caption", "shuffleTags"], value)} />
              </Row>
            </>
          )}
        </div>
      </details>
      )}
      {isAnimaLora && (
        <Row
          label="清理标签"
          description="每行一个，训练前从 caption 中移除；源文件不会改动。"
          errors={errorMap.get("dataset.caption.dropTokens")}
        >
          <textarea
            className="w-full max-w-2xl rounded-[4px] border border-input bg-background px-3 py-2 text-sm font-mono outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground/60"
            rows={4}
            placeholder={"1girl\nlooking at viewer\n2d, anime style"}
            value={(caption.dropTokens ?? []).join("\n")}
            onChange={(event) => {
              const lines = event.target.value
                .split("\n")
                .map((line) => line.trim())
                .filter((line) => line.length > 0)
              set(["dataset", "caption", "dropTokens"], lines)
            }}
          />
        </Row>
      )}
    </div>
  )
}
