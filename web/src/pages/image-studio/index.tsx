import { useSearchParams } from "react-router-dom"
import { DatasetManager } from "./components/dataset-manager"
import { DatasetDetail } from "./components/dataset-detail"

export { ImageStudioPage }

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const datasetPath = params.get("path") || ""

  if (!datasetPath) {
    return (
      <DatasetManager
        onOpen={(path) => {
          const next = new URLSearchParams()
          next.set("path", path)
          setParams(next)
        }}
      />
    )
  }

  return <DatasetDetail />
}
