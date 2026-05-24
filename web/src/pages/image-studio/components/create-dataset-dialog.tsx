import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface CreateDatasetDialogProps {
  onClose: () => void
  onCreate: (data: {
    name: string
    description?: string
    targetResolution?: string
    triggerWord?: string
  }) => void
  loading: boolean
}

export function CreateDatasetDialog({
  onClose,
  onCreate,
  loading,
}: CreateDatasetDialogProps) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [resolution, setResolution] = useState("")
  const [trigger, setTrigger] = useState("")

  const submit = () => {
    if (!name.trim() || loading) return
    onCreate({
      name: name.trim(),
      description: description || undefined,
      targetResolution: resolution || undefined,
      triggerWord: trigger || undefined,
    })
  }

  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>新建数据集</DialogTitle>
          <DialogDescription>
            会在工作区下创建同名目录，meta 信息保存到 dataset.toml。
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
          className="space-y-3"
        >
          <div className="space-y-1.5">
            <Label className="text-[11px]">
              名称 <span className="text-destructive">*</span>
            </Label>
            <Input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_character"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-[11px]">描述</Label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="角色 / 风格 / 用途备注"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-[11px]">目标分辨率</Label>
              <Input
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                placeholder="1024x1024"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[11px]">触发词</Label>
              <Input
                value={trigger}
                onChange={(e) => setTrigger(e.target.value)}
                placeholder="ohwx, sks, ..."
              />
            </div>
          </div>
        </form>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>
            取消
          </Button>
          <Button
            size="sm"
            onClick={submit}
            disabled={!name.trim() || loading}
          >
            {loading ? "创建中…" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
