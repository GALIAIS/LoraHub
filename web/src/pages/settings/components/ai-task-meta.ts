import type { AITaskId } from "@/lib/api"

export const TASK_LABELS: Record<AITaskId, string> = {
  "global.default": "默认 (兜底)",
  "tagging.assist": "VLM 打标",
  "caption.rewrite": "Caption 改写",
  "dataset.analyze": "数据集分析",
  "training.diagnose": "训练诊断",
  "error.diagnose": "错误自助",
  "quality.score": "图片质量评分",
  "trigger.suggest": "Trigger 生成",
  "config.recommend": "配置推荐",
}

export const TASK_DESCRIPTIONS: Record<AITaskId, string> = {
  "global.default": "其它任务未单独配置时的兜底路由",
  "tagging.assist": "图像工作台打标、ToriiGate/TAG+VLM 等视觉描述任务",
  "caption.rewrite": "把 WD14 标签改写为自然语言或统一格式",
  "dataset.analyze": "对扫描结果做诊断，检查 caption 长度、tag 分布等",
  "training.diagnose": "解读 loss/grad_norm 曲线并输出调整方案",
  "error.diagnose": "分析训练、安装失败并输出处理方案",
  "quality.score": "VLM 评估图片质量，输出分数与原因",
  "trigger.suggest": "根据数据集特征生成 trigger word 和模板",
  "config.recommend": "根据硬件、数据集、目标推荐训练配置",
}
