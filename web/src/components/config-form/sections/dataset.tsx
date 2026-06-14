import { memo } from "react"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import {
  IntInput,
  PathInput,
  ResolutionInput,
  Row,
  TextInput,
} from "../widgets"
import { DatasetSourceSelect } from "../dataset-source-select"
import { BucketSection, CaptionSection } from "./dataset-sections"

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
      <Row label="数据集" required errors={errorMap.get("dataset.source")}>
        <DatasetSourceSelect
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
          value={value.numRepeats}
          onChange={(v) => set(["dataset", "numRepeats"], v ?? 1)}
        />
      </Row>
      <Row
        label="conditioningDir"
        description="ControlNet / inpainting 条件图目录（kohya `--conditioning_data_dir`）。"
        errors={errorMap.get("dataset.conditioningDir")}
      >
        <PathInput
          value={value.conditioningDir ?? ""}
          onChange={(v) => set(["dataset", "conditioningDir"], v || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="regSource"
        description="DreamBooth 正则化数据集（kohya `--reg_data_dir`）。"
        errors={errorMap.get("dataset.regSource")}
      >
        <PathInput
          value={value.regSource ?? ""}
          onChange={(v) => set(["dataset", "regSource"], v || null)}
          placeholder="（可选）"
        />
      </Row>
      <Row
        label="frameBuckets"
        description="视频训练帧数桶，逗号分隔（例如 1,33,65）。默认 1（图像）。"
        errors={errorMap.get("dataset.frameBuckets")}
      >
        <TextInput
          className="w-64"
          value={(value.frameBuckets ?? [1]).join(",")}
          onChange={(s) => {
            const list = s
              .split(",")
              .map((x) => x.trim())
              .filter((x) => x.length > 0)
              .map((x) => parseInt(x, 10))
              .filter((n) => !Number.isNaN(n))
            set(["dataset", "frameBuckets"], list.length ? list : [1])
          }}
          placeholder="1"
        />
      </Row>

      <BucketSection bucket={bucket} set={set} errorMap={errorMap} />
      <CaptionSection caption={caption} set={set} errorMap={errorMap} />
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
                [...subsets, { path: "", numRepeats: 1 }],
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
            <Row label="numRepeats">
              <IntInput
                min={1}
                value={sub.numRepeats ?? 1}
                onChange={(n) =>
                  set(["dataset", "subsets", idx, "numRepeats"], n ?? 1)
                }
              />
            </Row>
            <Row label="maskPath" description="可选。掩码目录与图像目录布局一致。">
              <PathInput
                value={sub.maskPath ?? ""}
                onChange={(s) =>
                  set(["dataset", "subsets", idx, "maskPath"], s || null)
                }
                placeholder="（可选）"
              />
            </Row>
            <Row
              label="参考图目录"
              description="差异训练 (anima_lora conditioning) 用。每张目标图配对同名参考图;后端开关在 后端 → 差异训练。留空禁用。"
            >
              <PathInput
                value={sub.conditioningDataDir ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "subsets", idx, "conditioningDataDir"],
                    s || null,
                  )
                }
                placeholder="（可选）"
              />
            </Row>
            <Row label="captionPrefix">
              <TextInput
                className="w-64"
                value={sub.captionPrefix ?? ""}
                onChange={(s) =>
                  set(
                    ["dataset", "subsets", idx, "captionPrefix"],
                    s || null,
                  )
                }
                placeholder="（可选）"
              />
            </Row>
            <Row
              label="arBuckets"
              description="子集级 arBuckets（dp）。逗号分隔。"
            >
              <TextInput
                className="w-64"
                value={(sub.arBuckets ?? []).join(",")}
                onChange={(s) => {
                  const list = s
                    .split(",")
                    .map((x) => x.trim())
                    .filter((x) => x.length > 0)
                    .map((x) => parseFloat(x))
                    .filter((n) => !Number.isNaN(n))
                  set(
                    ["dataset", "subsets", idx, "arBuckets"],
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
