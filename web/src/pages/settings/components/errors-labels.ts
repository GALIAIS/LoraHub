import type { ErrorReportItem } from "@/lib/api"

export const SEVERITY_LABEL: Record<ErrorReportItem["severity"], string> = {
  fatal: "严重",
  error: "错误",
  warn: "警告",
  info: "信息",
}

export const SEVERITY_TONE: Record<ErrorReportItem["severity"], string> = {
  fatal: "text-destructive",
  error: "text-destructive",
  warn: "text-amber-700 dark:text-amber-400",
  info: "text-cyan-700 dark:text-cyan-400",
}

export const SOURCE_LABEL: Record<string, string> = {
  "backend.exception": "后端 · 未捕获异常",
  "backend.job": "后端 · 训练任务",
  "backend.lifespan": "后端 · 启动钩子",
  "backend.preflight": "后端 · 预检查",
  "backend.bootstrap": "后端 · 安装",
  "backend.update": "后端 · 自更新",
  "frontend.render": "前端 · 渲染崩溃",
  "frontend.runtime": "前端 · 运行时",
  "frontend.api": "前端 · API 调用",
  "user.report": "用户主动上报",
}
