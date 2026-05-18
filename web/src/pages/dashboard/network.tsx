import { memo, useMemo, useState } from "react"
import { Globe, Network, Wifi } from "lucide-react"
import type {
  NetworkInterfaceStats,
  PublicIpInfo,
  TcpConnectionStats,
} from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import { formatRate } from "./_shared"

export const NetworkInterfacesCard = memo(function NetworkInterfacesCard({
  interfaces,
}: {
  interfaces: NetworkInterfaceStats[]
}) {
  const [showAll, setShowAll] = useState(false)
  // Loopback / virtual NICs add noise on most workstations; hide them by
  // default and let the user opt in with the toggle. We keep them in the
  // dataset so kind counts in the badge are accurate.
  const filtered = useMemo(() => {
    if (showAll) return interfaces
    return interfaces.filter((nic) => nic.kind !== "loopback" && nic.kind !== "virtual")
  }, [interfaces, showAll])
  const hiddenCount = interfaces.length - filtered.length
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Network className="size-4 text-muted-foreground" />
              网络接口
            </CardTitle>
            <CardDescription className="text-xs">
              {showAll
                ? `共 ${interfaces.length} 张网卡（含回环 / 虚拟）。`
                : `显示 ${filtered.length} 张${
                    hiddenCount > 0 ? `（已隐藏 ${hiddenCount} 张回环 / 虚拟）` : ""
                  }`}
            </CardDescription>
          </div>
          <button
            type="button"
            onClick={() => setShowAll((v) => !v)}
            className="rounded-[2px] border border-border/80 bg-background/78 px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] text-foreground transition-colors hover:bg-muted"
          >
            {showAll ? "仅物理" : "显示全部"}
          </button>
        </div>
      </CardHeader>
      <CardContent>
        {filtered.length === 0 ? (
          <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/20 px-4 py-6 text-center text-xs text-muted-foreground">
            当前没有可显示的接口
          </div>
        ) : (
          <div className="max-h-[16rem] overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 bg-card z-10">
              <TableRow>
                <TableHead>名称</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>IPv4</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[8ch]">链路</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[10ch]">入</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[10ch]">出</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((nic) => {
                const ipv4 = nic.addresses.find(
                  (a) => a.family === "AF_INET" || a.family === "ipv4" || a.family === "IPv4",
                )
                return (
                  <TableRow key={nic.name} className={cn(!nic.is_up && "opacity-60")}>
                    <TableCell className="font-mono text-xs">{nic.name}</TableCell>
                    <TableCell>
                      <NicKindBadge kind={nic.kind} isUp={nic.is_up} />
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {ipv4 ? ipv4.address : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                      {typeof nic.speed_mbps === "number" && nic.speed_mbps > 0
                        ? `${nic.speed_mbps} Mb/s`
                        : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                      {formatRate(nic.bytes_recv_per_sec)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                      {formatRate(nic.bytes_sent_per_sec)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
})

const NicKindBadge = memo(function NicKindBadge({
  kind,
  isUp,
}: {
  kind: NetworkInterfaceStats["kind"]
  isUp: boolean
}) {
  // physical=primary, wireless=secondary, virtual=outline, loopback=ghost.
  const variant = ({
    physical: "default",
    wireless: "secondary",
    virtual: "outline",
    loopback: "ghost",
  } as const)[kind] ?? "outline"
  const label = {
    physical: "Physical",
    wireless: "Wireless",
    virtual: "Virtual",
    loopback: "Loopback",
  }[kind] ?? kind
  return (
    <div className="flex items-center gap-1.5">
      <Badge variant={variant} className="rounded-[2px] gap-1">
        {kind === "wireless" ? <Wifi className="size-3" /> : null}
        {label}
      </Badge>
      {!isUp && (
        <Badge variant="outline" className="rounded-[2px] text-[9px]">
          DOWN
        </Badge>
      )}
    </div>
  )
})

export const NetworkSummaryCard = memo(function NetworkSummaryCard({
  tcp,
  publicIp,
}: {
  tcp: TcpConnectionStats | null
  publicIp: PublicIpInfo | null
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Globe className="size-4 text-muted-foreground" />
          网络概览
        </CardTitle>
        <CardDescription className="text-xs">
          公网 IP 与 TCP 连接状态聚合。
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              公网 IP
            </div>
            {publicIp ? (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-sm">
                  {publicIp.ip ?? "—"}
                </span>
                <PublicIpSourceBadge source={publicIp.source} />
                {publicIp.fetched_at > 0 && (
                  <span
                    className="text-[10px] text-muted-foreground tabular-nums"
                    title={new Date(publicIp.fetched_at * 1000).toLocaleString()}
                  >
                    {new Date(publicIp.fetched_at * 1000).toLocaleTimeString()}
                  </span>
                )}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">—</div>
            )}
          </div>
          <div className="space-y-2">
            <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              TCP 连接
            </div>
            {tcp ? (
              <div className="grid grid-cols-3 gap-x-3 gap-y-1.5 text-xs font-mono tabular-nums">
                <TcpStat label="已建立" value={tcp.established} />
                <TcpStat label="监听" value={tcp.listen} />
                <TcpStat label="TIME_WAIT" value={tcp.time_wait} />
                <TcpStat label="CLOSE_WAIT" value={tcp.close_wait} />
                <TcpStat label="其它" value={tcp.other} />
                <TcpStat label="合计" value={tcp.total} highlight />
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">—</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
})

const TcpStat = memo(function TcpStat({
  label,
  value,
  highlight,
}: {
  label: string
  value: number
  highlight?: boolean
}) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "tabular-nums",
          highlight ? "text-foreground font-semibold" : "text-foreground/85",
        )}
      >
        {value}
      </div>
    </div>
  )
})

const PublicIpSourceBadge = memo(function PublicIpSourceBadge({
  source,
}: {
  source: PublicIpInfo["source"]
}) {
  const map: Record<string, { label: string; variant: "default" | "secondary" | "outline" | "destructive" }> = {
    "ip.sb": { label: "ip.sb", variant: "secondary" },
    "ipinfo.io": { label: "ipinfo.io", variant: "secondary" },
    cached: { label: "cached", variant: "outline" },
    unreachable: { label: "unreachable", variant: "destructive" },
  }
  const meta = map[source] ?? { label: String(source), variant: "outline" as const }
  return (
    <Badge variant={meta.variant} className="rounded-[2px] gap-1 font-mono lowercase">
      {meta.label}
    </Badge>
  )
})
