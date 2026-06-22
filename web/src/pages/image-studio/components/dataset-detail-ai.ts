import {
  startQualitySession,
  startSmartCaptionSession,
  startTaggingSession,
  startTriggerWordsSession,
} from "@/lib/api"
import { addTask } from "@/lib/studio-task-store"
import type { AiBulkTab } from "./types"

export async function startDatasetAiBulkTask({
  tab,
  params,
  path,
  recursive,
}: {
  tab: AiBulkTab
  params: Record<string, unknown>
  path: string
  recursive: boolean
}) {
  const taskPath = (params.path as string) || path

  switch (tab) {
    case "smart-caption": {
      const captionSource =
        (params.captionSource as "vlm" | "tags" | undefined) ?? "vlm"
      const submit = await startSmartCaptionSession({
        path: taskPath,
        recursive,
        device: params.device as string,
        mergeStrategy: params.mergeStrategy as string,
        captionMode: params.captionMode as "general" | "style" | "character",
        captionSource,
        triggerWord: params.triggerWord as string | undefined,
        stripStyleTags: params.stripStyleTags as boolean | undefined,
        skipExisting: params.skipExisting as boolean | undefined,
      })
      addTask({
        id: submit.session_id,
        kind: "smart-caption",
        datasetPath: taskPath,
        label:
          captionSource === "tags"
            ? "智能标注（WD14 + LLM 文本模式）"
            : "智能标注（WD14 + VLM 视觉模式）",
        total: submit.total,
      })
      return
    }
    case "wd14": {
      const session = await startTaggingSession({
        path: taskPath,
        tagger: (params.model_id as string)?.startsWith("joy")
          ? "joytag"
          : "wd14",
        model_id: params.model_id as string,
        general: params.general as number,
        character: params.character as number,
        device: params.device as string,
        overwrite: params.overwrite as boolean,
        recursive,
      })
      addTask({
        id: session.session_id,
        kind: "wd14",
        datasetPath: taskPath,
        label: "WD14 标注",
      })
      return
    }
    case "quality-score": {
      const submit = await startQualitySession({
        path: taskPath,
        recursive,
        skipScored: params.skipScored as boolean | undefined,
      })
      addTask({
        id: submit.session_id,
        kind: "quality-score",
        datasetPath: taskPath,
        label: submit.skipped
          ? `质量评分（跳过 ${submit.skipped} 已评分）`
          : "质量评分",
        total: submit.total,
      })
      return
    }
    case "trigger-words": {
      const submit = await startTriggerWordsSession({
        path: taskPath,
        recursive,
        skipAnalyzed: params.skipAnalyzed as boolean | undefined,
      })
      addTask({
        id: submit.session_id,
        kind: "trigger-words",
        datasetPath: taskPath,
        label: submit.skipped
          ? `分析触发词（跳过 ${submit.skipped} 已分析）`
          : "分析触发词",
        total: submit.total,
      })
      return
    }
  }
}
