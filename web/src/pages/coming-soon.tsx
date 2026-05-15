import { Construction } from "lucide-react"

export function ComingSoonPage({ title }: { title: string }) {
  return (
    <div className="h-screen grid place-items-center px-8">
      <div className="text-center max-w-md">
        <Construction className="size-10 mx-auto text-muted-foreground/50" />
        <h1 className="mt-3 text-xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Coming in a future v0.2.x release. For now use the CLI:{" "}
          <code className="text-foreground">lorahub --help</code>
        </p>
      </div>
    </div>
  )
}
