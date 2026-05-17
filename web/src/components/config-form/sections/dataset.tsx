import { memo } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  CAPTION_STRATEGY_OPTIONS,
  MAX_TOKEN_LENGTH_OPTIONS,
  RESIZE_INTERPOLATION_OPTIONS,
} from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  PathInput,
  ResolutionInput,
  Row,
  TextInput,
  ToggleSwitch,
} from "../widgets"

export const DatasetFields = memo(function DatasetFields({
  value,
  set,
  errorMap,
}: {
  value: ConfigFormValue["dataset"]
  set: Setter
  errorMap: ErrorMap
}) {
  const bucket = value.bucket ?? {}
  const caption = value.caption ?? {}
  const subsets = value.subsets ?? []
  return (
    <>
      <Row label="数据集路径" required errors={errorMap.get("dataset.source")}>
        <PathInput
          value={value.source}
          onChange={(v) => set(["dataset", "source"], v)}
          placeholder="./datasets/my_character"
        />
      </Row>
      <Row label="分辨率" errors={errorMap.get("dataset.resolution")}>
        <ResolutionInput
          value={value.resolution}
          onChange={(v) => set(["dataset", "resolution"], v)}
        />
      </Row>
      <Row label="重复次数" description="每个回合中每张图被使用的次数。">
        <IntInput
          min={1}
          value={value.num_repeats}
          onChange={(v) => set(["dataset", "num_repeats"], v ?? 1)}
        />
      </Row>
      <Row
        label="conditioning_dir"
        description="ControlNet / inpainting 条件图目录（kohya `--conditioning_data_dir`）。"
        errors={errorMap.get("dataset.conditioning_dir")}
      >
        <PathInput
          value={value.conditioning_dir ?? ""}
          onChange={(v) => set(["dataset", "conditioning_dir"], v || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="reg_source"
        description="DreamBooth 正则化数据集（kohya `--reg_data_dir`）。"
        errors={errorMap.get("dataset.reg_source")}
      >
        <PathInput
          value={value.reg_source ?? ""}
          onChange={(v) => set(["dataset", "reg_source"], v || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="frame_buckets"
        description="视频训练帧数桶，逗号分隔（例如 1,33,65）。默认 1（图像）。"
        errors={errorMap.get("dataset.frame_buckets")}
      >
        <TextInput
          className="w-64"
          value={(value.frame_buckets ?? [1]).join(",")}
          onChange={(s) => {
            const list = s
              .split(",")
              .map((x) => x.trim())
              .filter((x) => x.length > 0)
              .map((x) => parseInt(x, 10))
              .filter((n) => !Number.isNaN(n))
            set(["dataset", "frame_buckets"], list.length ? list : [1])
          }}
          placeholder="1"
        />
      </Row>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
            分桶 Bucket
          </span>
          <ToggleSwitch
            checked={bucket.enabled ?? true}
            onCheckedChange={(v) => set(["dataset", "bucket", "enabled"], v)}
          />
        </div>
        {bucket.enabled !== false && (
          <>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <Label className="text-[11px] text-muted-foreground">最小边</Label>
                <IntInput
                  min={64}
                  value={bucket.min ?? 256}
                  onChange={(v) => set(["dataset", "bucket", "min"], v ?? 256)}
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">最大边</Label>
                <IntInput
                  min={64}
                  value={bucket.max ?? 2048}
                  onChange={(v) => set(["dataset", "bucket", "max"], v ?? 2048)}
                />
              </div>
              <div>
                <Label className="text-[11px] text-muted-foreground">步长</Label>
                <IntInput
                  min={8}
                  value={bucket.step ?? 64}
                  onChange={(v) => set(["dataset", "bucket", "step"], v ?? 64)}
                />
              </div>
            </div>
            <Row
              label="no_upscale"
              description="不要把小图放大到桶尺寸（kohya `--bucket_no_upscale`）。"
            >
              <ToggleSwitch
                checked={bucket.no_upscale ?? false}
                onCheckedChange={(v) =>
                  set(["dataset", "bucket", "no_upscale"], v)
                }
              />
            </Row>
            <Row
              label="skip_image_resolution"
              description="跳过图像分辨率合理性检查（kohya `--skip_image_resolution`）。"
            >
              <ToggleSwitch
                checked={bucket.skip_image_resolution ?? false}
                onCheckedChange={(v) =>
                  set(["dataset", "bucket", "skip_image_resolution"], v)
                }
              />
            </Row>
            <Row label="resize_interpolation" description="PIL 重采样核。">
              <EnumSelect
                value={bucket.resize_interpolation ?? ""}
                onChange={(v) =>
                  set(["dataset", "bucket", "resize_interpolation"], v || null)
                }
                options={RESIZE_INTERPOLATION_OPTIONS}
              />
            </Row>
            <Row
              label="ar_buckets"
              description="dp 显式宽高比列表，逗号分隔（覆盖 min/max/num）。"
              errors={errorMap.get("dataset.bucket.ar_buckets")}
            >
              <TextInput
                className="w-64"
                value={(bucket.ar_buckets ?? []).join(",")}
                onChange={(s) => {
                  const list = s
                    .split(",")
                    .map((x) => x.trim())
                    .filter((x) => x.length > 0)
                    .map((x) => parseFloat(x))
                    .filter((n) => !Number.isNaN(n))
                  set(
                    ["dataset", "bucket", "ar_buckets"],
                    list.length ? list : null,
                  )
                }}
                placeholder="（默认）"
              />
            </Row>
          </>
        )}
      </div>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          标注 Caption
        </div>
        <Row label="策略">
          <EnumSelect
            value={caption.strategy ?? "tag_file"}
            onChange={(v) => set(["dataset", "caption", "strategy"], v)}
            options={CAPTION_STRATEGY_OPTIONS}
          />
        </Row>
        <Row label="标注扩展名" description=".txt 是 kohya 默认值。">
          <Input
            value={caption.ext ?? ".txt"}
            className="font-mono w-32"
            onChange={(e) => set(["dataset", "caption", "ext"], e.target.value)}
          />
        </Row>
        <Row label="打乱标签" description="每步随机打乱以逗号分隔的标签。">
          <ToggleSwitch
            checked={caption.shuffle ?? true}
            onCheckedChange={(v) => set(["dataset", "caption", "shuffle"], v)}
          />
        </Row>
        <Row
          label="丢弃概率"
          description="0-1 之间，每步随机丢弃单个标签的概率。"
          errors={errorMap.get("dataset.caption.drop_rate")}
        >
          <FloatInput
            step={0.05}
            value={caption.drop_rate ?? 0}
            onChange={(v) => set(["dataset", "caption", "drop_rate"], v ?? 0)}
          />
        </Row>

        <details className="rounded-[4px] border border-border/40 bg-muted/10 px-3 py-2 group">
          <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.18em]">
            高级 Caption 选项
          </summary>
          <div className="mt-3 space-y-3.5">
            <Row
              label="dropout_every_n_epochs"
              description="每 N 回合做一次 caption dropout（与 drop_rate 不同）。"
              errors={errorMap.get("dataset.caption.dropout_every_n_epochs")}
            >
              <IntInput
                min={0}
                value={caption.dropout_every_n_epochs ?? 0}
                onChange={(n) =>
                  set(["dataset", "caption", "dropout_every_n_epochs"], n ?? 0)
                }
              />
            </Row>
            <Row
              label="tag_dropout_rate"
              description="单个 tag 维度的 dropout（0..1）。"
              errors={errorMap.get("dataset.caption.tag_dropout_rate")}
            >
              <FloatInput
                step={0.05}
                value={caption.tag_dropout_rate ?? 0}
                onChange={(n) =>
                  set(["dataset", "caption", "tag_dropout_rate"], n ?? 0)
                }
              />
            </Row>
            <Row
              label="keep_tokens"
              description="前 N 个 tag 永不被打乱（典型用法：触发词锁在 0 位）。"
              errors={errorMap.get("dataset.caption.keep_tokens")}
            >
              <IntInput
                min={0}
                value={caption.keep_tokens ?? 0}
                onChange={(n) =>
                  set(["dataset", "caption", "keep_tokens"], n ?? 0)
                }
              />
            </Row>
            <Row
              label="keep_tokens_separator"
              description="自定义 keep_tokens 与可洗牌段之间的分隔符。"
            >
              <TextInput
                className="w-32"
                value={caption.keep_tokens_separator ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "caption", "keep_tokens_separator"],
                    s || null,
                  )
                }
                placeholder=","
              />
            </Row>
            <Row label="secondary_separator" description="kohya 二级分隔符。">
              <TextInput
                className="w-32"
                value={caption.secondary_separator ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "caption", "secondary_separator"],
                    s || null,
                  )
                }
                placeholder="（可选）"
              />
            </Row>
            <Row
              label="enable_wildcard"
              description="启用 caption 中 `{a|b|c}` 通配符（kohya `--enable_wildcard`）。"
            >
              <ToggleSwitch
                checked={caption.enable_wildcard ?? false}
                onCheckedChange={(b) =>
                  set(["dataset", "caption", "enable_wildcard"], b)
                }
              />
            </Row>
            <Row label="prefix" description="所有 caption 之前追加的前缀。">
              <TextInput
                className="w-64"
                value={caption.prefix ?? ""}
                onChange={(s) =>
                  set(["dataset", "caption", "prefix"], s || null)
                }
                placeholder="（可选）"
              />
            </Row>
            <Row label="suffix" description="所有 caption 之后追加的后缀。">
              <TextInput
                className="w-64"
                value={caption.suffix ?? ""}
                onChange={(s) =>
                  set(["dataset", "caption", "suffix"], s || null)
                }
                placeholder="（可选）"
              />
            </Row>
            <Row
              label="max_token_length"
              description="kohya `--max_token_length`：75 / 150 / 225。"
            >
              <EnumSelect
                value={
                  caption.max_token_length === null ||
                  caption.max_token_length === undefined
                    ? ""
                    : String(caption.max_token_length)
                }
                onChange={(s) =>
                  set(
                    ["dataset", "caption", "max_token_length"],
                    s ? parseInt(s, 10) : null,
                  )
                }
                options={MAX_TOKEN_LENGTH_OPTIONS}
              />
            </Row>
            <Row
              label="token_warmup_min"
              description="token warmup 最小 tag 数（kohya）。"
              errors={errorMap.get("dataset.caption.token_warmup_min")}
            >
              <IntInput
                min={1}
                value={caption.token_warmup_min ?? null}
                onChange={(n) =>
                  set(["dataset", "caption", "token_warmup_min"], n)
                }
                placeholder="（默认）"
              />
            </Row>
            <Row
              label="token_warmup_step"
              description="token warmup 步数。"
              errors={errorMap.get("dataset.caption.token_warmup_step")}
            >
              <FloatInput
                step={0.1}
                value={caption.token_warmup_step ?? null}
                onChange={(n) =>
                  set(["dataset", "caption", "token_warmup_step"], n)
                }
                placeholder="（默认）"
              />
            </Row>
            <Row
              label="weighted"
              description="启用 weighted captions（lpw 风格 `(token:1.5)`）。"
            >
              <ToggleSwitch
                checked={caption.weighted ?? false}
                onCheckedChange={(b) =>
                  set(["dataset", "caption", "weighted"], b)
                }
              />
            </Row>
            <Row
              label="shuffle_delimiter"
              description="dp tag shuffle 分隔符（默认 `, `）。"
            >
              <TextInput
                className="w-32"
                value={caption.shuffle_delimiter ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "caption", "shuffle_delimiter"],
                    s || null,
                  )
                }
                placeholder="（默认）"
              />
            </Row>
            <Row
              label="shuffle_tags"
              description="dp legacy 整 caption 打乱模式。"
            >
              <ToggleSwitch
                checked={caption.shuffle_tags ?? false}
                onCheckedChange={(b) =>
                  set(["dataset", "caption", "shuffle_tags"], b)
                }
              />
            </Row>
          </div>
        </details>
      </div>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
            子集 Subsets（覆盖 source）
          </span>
          <button
            type="button"
            className="text-[11px] px-2 py-1 rounded-[4px] border border-border/60 hover:bg-muted/40"
            onClick={() =>
              set(
                ["dataset", "subsets"],
                [...subsets, { path: "", num_repeats: 1 }],
              )
            }
          >
            + 添加子集
          </button>
        </div>
        {subsets.length === 0 && (
          <div className="text-[11px] text-muted-foreground/80">
            未填则使用上方单一 source；填写则覆盖（dp 每条对应一个 [[directory]]，kohya 合成等价的 dataset toml）。
          </div>
        )}
        {subsets.map((sub, idx) => (
          <div
            key={idx}
            className="rounded-[4px] border border-border/40 p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-muted-foreground">
                #{idx + 1}
              </span>
              <button
                type="button"
                className="text-[11px] text-destructive hover:underline"
                onClick={() => {
                  const next = subsets.slice()
                  next.splice(idx, 1)
                  set(["dataset", "subsets"], next)
                }}
              >
                删除
              </button>
            </div>
            <Row
              label="path"
              required
              errors={errorMap.get(`dataset.subsets.${idx}.path`)}
            >
              <PathInput
                value={sub.path ?? ""}
                onChange={(s) => set(["dataset", "subsets", idx, "path"], s)}
                placeholder="./datasets/subset"
              />
            </Row>
            <Row label="num_repeats">
              <IntInput
                min={1}
                value={sub.num_repeats ?? 1}
                onChange={(n) =>
                  set(["dataset", "subsets", idx, "num_repeats"], n ?? 1)
                }
              />
            </Row>
            <Row label="mask_path" description="可选。掩码目录与图像目录布局一致。">
              <PathInput
                value={sub.mask_path ?? ""}
                onChange={(s) =>
                  set(["dataset", "subsets", idx, "mask_path"], s || null)
                }
                placeholder="（可选）"
              />
            </Row>
            <Row label="caption_prefix">
              <TextInput
                className="w-64"
                value={sub.caption_prefix ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "subsets", idx, "caption_prefix"],
                    s || null,
                  )
                }
                placeholder="（可选）"
              />
            </Row>
            <Row
              label="ar_buckets"
              description="子集级 ar_buckets（dp）。逗号分隔。"
            >
              <TextInput
                className="w-64"
                value={(sub.ar_buckets ?? []).join(",")}
                onChange={(s) => {
                  const list = s
                    .split(",")
                    .map((x) => x.trim())
                    .filter((x) => x.length > 0)
                    .map((x) => parseFloat(x))
                    .filter((n) => !Number.isNaN(n))
                  set(
                    ["dataset", "subsets", idx, "ar_buckets"],
                    list.length ? list : null,
                  )
                }}
                placeholder="（默认）"
              />
            </Row>
          </div>
        ))}
      </div>
    </>
  )
})
