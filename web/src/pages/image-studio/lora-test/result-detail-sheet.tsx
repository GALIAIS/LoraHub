/**
 * 结果详情 Sheet — 点开单张图片查看完整参数并可复制。
 */
import { Copy, Download } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Param } from "./fields"
import type { ResultImage } from "./types"

export function ResultSheet({
  sessionId,
  image,
  onClose,
}: {
  sessionId: string
  image: ResultImage | null
  onClose: () => void
}) {
  return (
    <Sheet open={image !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-3xl" side="right">
        <SheetHeader>
          <SheetTitle>生成结果</SheetTitle>
          <SheetDescription>
            图片和完整参数，可复制到下一轮测试。
          </SheetDescription>
        </SheetHeader>
        {image && (
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_18rem]">
              <div className="grid min-h-[24rem] place-items-center rounded-[6px] border bg-background">
                <img
                  src={api.loraTestResultFileUrl(sessionId, image.path)}
                  alt={`seed ${image.seed}`}
                  className="max-h-[70vh] max-w-full object-contain"
                />
              </div>
              <div className="flex flex-col gap-3 text-xs">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    navigator.clipboard.writeText(JSON.stringify(image, null, 2))
                  }
                >
                  <Copy />
                  复制参数
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    window.open(
                      api.loraTestResultFileUrl(sessionId, image.path),
                      "_blank",
                      "noopener,noreferrer",
                    )
                  }
                >
                  <Download />
                  打开图片
                </Button>
                <Param label="seed" value={String(image.seed)} />
                <Param label="尺寸" value={`${image.width} x ${image.height}`} />
                <Param label="steps / cfg" value={`${image.steps} / ${image.cfg}`} />
                <Param label="sampler" value={image.sampler} />
                <Param
                  label="LoRA"
                  value={
                    image.loras?.length
                      ? image.loras
                          .map((item) => `${item.checkpoint_name} x ${item.weight}`)
                          .join("\n")
                      : String(image.lora_weight)
                  }
                  block
                />
                <Param label="checkpoint" value={image.checkpoint_path} />
                <Param label="prompt" value={image.prompt} block />
                <Param label="negative" value={image.negative_prompt || "-"} block />
              </div>
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
