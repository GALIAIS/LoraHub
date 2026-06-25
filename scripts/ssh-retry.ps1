param(
  [string]$Profile = "v100",
  [string]$SshFile = "E:\WorkSpace\Lora Scripts\ssh.txt",
  [string]$Command = "echo ok",
  [int]$Retries = 12,
  [int]$DelaySeconds = 8,
  [string]$KeyFile = ".lorahub\ssh\v100_ed25519",
  [switch]$PasswordAuth
)

$ErrorActionPreference = "Stop"

function Get-SshTarget {
  param([string]$Path, [string]$Name)

  $lines = Get-Content -LiteralPath $Path
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i].Trim() -ne $Name) { continue }
    for ($j = $i + 1; $j -lt $lines.Count; $j++) {
      $line = $lines[$j].Trim()
      if ($line -match '^ssh\s+') { return $line }
    }
  }

  if ($Name -eq "default") {
    foreach ($line in $lines) {
      $line = $line.Trim()
      if ($line -match '^ssh\s+') { return $line }
    }
  }

  throw "SSH profile '$Name' not found in $Path"
}

$sshLine = Get-SshTarget -Path $SshFile -Name $Profile
$target = $sshLine -replace '^ssh\s+', ''
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command))
$remoteCommand = "printf '%s' '$encodedCommand' | base64 -d | bash"
$argsPrefix = @(
  "-o", "ConnectTimeout=10",
  "-o", "ConnectionAttempts=1",
  "-o", "ServerAliveInterval=15",
  "-o", "ServerAliveCountMax=3",
  "-o", "TCPKeepAlive=yes",
  "-o", "StrictHostKeyChecking=accept-new"
)

if (!$PasswordAuth -and (Test-Path -LiteralPath $KeyFile)) {
  $argsPrefix += @(
    "-i", (Resolve-Path -LiteralPath $KeyFile).Path,
    "-o", "PreferredAuthentications=publickey",
    "-o", "BatchMode=yes"
  )
} else {
  $argsPrefix += @(
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no"
  )
}

$argsPrefix += ($target -split '\s+')

for ($attempt = 1; $attempt -le $Retries; $attempt++) {
  Write-Host "ssh attempt $attempt/${Retries}: ssh $target"
  & ssh @argsPrefix -- $remoteCommand
  if ($LASTEXITCODE -eq 0) { exit 0 }
  if ($attempt -lt $Retries) {
    Start-Sleep -Seconds ([Math]::Min($DelaySeconds * $attempt, 60))
  }
}

exit $LASTEXITCODE
