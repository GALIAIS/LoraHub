<#
.SYNOPSIS
  Quick dev launcher: API + Vite in two independent windows.

.DESCRIPTION
  The richer scripts/launch.ps1 tries to track service lifecycles, which
  bumps into ShellExecute restrictions on .cmd shims (npm.cmd) and the
  short-lived cmd.exe parent problem. This wrapper avoids the whole
  rabbit hole: it spawns each service in a detached new console window,
  which is what most Windows devs actually want anyway — you can tail
  the logs visually and Ctrl+C either side independently.

.PARAMETER ApiPort
  Default 18765.

.PARAMETER WebPort
  Default 6006.
#>

[CmdletBinding()]
param(
  [int]$ApiPort = 18765,
  [int]$WebPort = 6006
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

function Test-PortFree([int]$port) {
  try {
    $listener = [System.Net.Sockets.TcpListener]::new(
      [System.Net.IPAddress]::Loopback, $port
    )
    $listener.Start()
    $listener.Stop()
    return $true
  } catch {
    return $false
  }
}

if (-not (Test-PortFree $ApiPort)) {
  throw "Port $ApiPort is busy. Stop the holder or pass -ApiPort <other>."
}
if (-not (Test-PortFree $WebPort)) {
  throw "Port $WebPort is busy. Stop the holder or pass -WebPort <other>."
}

# Resolve python: prefer project venv, fall back to PATH.
$python = $null
foreach ($candidate in @(
    (Join-Path $Root '.venv\Scripts\python.exe'),
    (Join-Path $Root '.venv\bin\python')
  )) {
  if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { $python = $cmd.Source }
}
if (-not $python) { throw 'No Python interpreter found.' }

Write-Host "[lorahub] Python: $python" -ForegroundColor Cyan
Write-Host "[lorahub] API window -> http://127.0.0.1:$ApiPort" -ForegroundColor Cyan
Write-Host "[lorahub] Web window -> http://localhost:$WebPort  (proxying /api -> http://127.0.0.1:$ApiPort)" -ForegroundColor Cyan

# Each service goes into its own console window. -PassThru returns the
# launching process; on Windows that ends up being the new conhost +
# uvicorn/npm chain, but we don't care here — we hand control back to
# the user and let them close windows when done.
Start-Process -FilePath $python `
  -ArgumentList @('-m', 'uvicorn', 'lorahub.api.app:app',
                  '--host', '127.0.0.1', '--port', "$ApiPort",
                  '--log-level', 'info') `
  -WorkingDirectory $Root `
  -WindowStyle Normal | Out-Null

# Vite picks up LORAHUB_API_TARGET via vite.config.ts to forward /api.
$env:LORAHUB_API_TARGET = "http://127.0.0.1:$ApiPort"

# `npm.cmd` is a batch shim. Routing through cmd.exe with /k keeps the
# console alive so the user sees Vite logs and can Ctrl+C cleanly. The
# /k command line stays as one argv[1] string so cmd doesn't fight us
# over quoting around `--`.
$cmdLine = "cd /d `"$(Join-Path $Root 'web')`" && set LORAHUB_API_TARGET=$env:LORAHUB_API_TARGET && npm run dev -- --host 127.0.0.1 --port $WebPort"
Start-Process -FilePath 'cmd.exe' `
  -ArgumentList @('/k', $cmdLine) `
  -WindowStyle Normal | Out-Null

Write-Host "[lorahub] Both windows launched. Close them to stop." -ForegroundColor Green
Write-Host "[lorahub] Health checks (give Vite ~10s to warm up):" -ForegroundColor Cyan
Write-Host "    curl http://127.0.0.1:$ApiPort/api/health"
Write-Host "    curl http://127.0.0.1:$WebPort/"
