/**
 * DataLoaderConfig editor — DataLoader / cache pipeline knobs.
 *
 * Maps to kohya: --max_data_loader_n_workers,
 * --persistent_data_loader_workers, --vae_batch_size, --text_encoder_batch_size
 * and dp's caching_batch_size / map_num_proc.
 */
import { memo } from "react"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { IntInput, Row, ToggleSwitch } from "../widgets"

export const DataLoaderFields = memo(function DataLoaderFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["dataloader"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row
        label="num_workers"
        description="DataLoader worker 数；0 表示主进程。"
        errors={errorMap.get("dataloader.num_workers")}
      >
        <IntInput
          min={0}
          value={v.num_workers ?? 8}
          onChange={(n) => set(["dataloader", "num_workers"], n ?? 0)}
        />
      </Row>
      <Row
        label="persistent_workers"
        description="跨 epoch 保持 worker 进程，省启动开销。"
      >
        <ToggleSwitch
          checked={v.persistent_workers ?? false}
          onCheckedChange={(b) => set(["dataloader", "persistent_workers"], b)}
        />
      </Row>
      <Row
        label="vae_batch_size"
        description="VAE 编码阶段的批大小。"
        errors={errorMap.get("dataloader.vae_batch_size")}
      >
        <IntInput
          min={1}
          value={v.vae_batch_size ?? 1}
          onChange={(n) => set(["dataloader", "vae_batch_size"], n ?? 1)}
        />
      </Row>
      <Row
        label="text_encoder_batch_size"
        description="文本编码阶段的批大小（kohya）。留空使用默认。"
        errors={errorMap.get("dataloader.text_encoder_batch_size")}
      >
        <IntInput
          min={1}
          value={v.text_encoder_batch_size ?? null}
          onChange={(n) => set(["dataloader", "text_encoder_batch_size"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="cache_shuffle_num"
        description="预缓存阶段打乱样本数；0 保持原顺序。"
        errors={errorMap.get("dataloader.cache_shuffle_num")}
      >
        <IntInput
          min={0}
          value={v.cache_shuffle_num ?? 0}
          onChange={(n) => set(["dataloader", "cache_shuffle_num"], n ?? 0)}
        />
      </Row>
      <Row
        label="map_num_proc"
        description="dp 数据集 map 的并行进程数。留空使用默认。"
        errors={errorMap.get("dataloader.map_num_proc")}
      >
        <IntInput
          min={1}
          value={v.map_num_proc ?? null}
          onChange={(n) => set(["dataloader", "map_num_proc"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})
