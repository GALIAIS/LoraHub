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
        label="numWorkers"
        description="DataLoader worker 数；0 表示主进程。"
        errors={errorMap.get("dataloader.numWorkers")}
      >
        <IntInput
          min={0}
          value={v.numWorkers ?? 8}
          onChange={(n) => set(["dataloader", "numWorkers"], n ?? 0)}
        />
      </Row>
      <Row
        label="persistentWorkers"
        description="跨 epoch 保持 worker 进程，省启动开销。"
      >
        <ToggleSwitch
          checked={v.persistentWorkers ?? false}
          onCheckedChange={(b) => set(["dataloader", "persistentWorkers"], b)}
        />
      </Row>
      <Row
        label="vaeBatchSize"
        description="VAE 编码阶段的批大小。"
        errors={errorMap.get("dataloader.vaeBatchSize")}
      >
        <IntInput
          min={1}
          value={v.vaeBatchSize ?? 1}
          onChange={(n) => set(["dataloader", "vaeBatchSize"], n ?? 1)}
        />
      </Row>
      <Row
        label="textEncoderBatchSize"
        description="文本编码阶段的批大小（kohya）。留空使用默认。"
        errors={errorMap.get("dataloader.textEncoderBatchSize")}
      >
        <IntInput
          min={1}
          value={v.textEncoderBatchSize ?? null}
          onChange={(n) => set(["dataloader", "textEncoderBatchSize"], n)}
          placeholder="（默认）"
        />
      </Row>
      <Row
        label="cacheShuffleNum"
        description="预缓存阶段打乱样本数；0 保持原顺序。"
        errors={errorMap.get("dataloader.cacheShuffleNum")}
      >
        <IntInput
          min={0}
          value={v.cacheShuffleNum ?? 0}
          onChange={(n) => set(["dataloader", "cacheShuffleNum"], n ?? 0)}
        />
      </Row>
      <Row
        label="mapNumProc"
        description="dp 数据集 map 的并行进程数。留空使用默认。"
        errors={errorMap.get("dataloader.mapNumProc")}
      >
        <IntInput
          min={1}
          value={v.mapNumProc ?? null}
          onChange={(n) => set(["dataloader", "mapNumProc"], n)}
          placeholder="（默认）"
        />
      </Row>
    </>
  )
})
