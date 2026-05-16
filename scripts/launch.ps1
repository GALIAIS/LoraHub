<#
.SYNOPSIS
  LoraHub launcher (Windows).

.DESCRIPTION
  Brings up the FastAPI backend (Uvicorn) and the React dev server,
  or just one of them. Detects a project virtualenv, prepares Python
  dependencies the first time it sees them, and prints a single line
  per service so you can tell at a glance what's running.

.PARAMETER Mode
  dev     Backend + frontend dev server (default).
  prod    Backend only — serves the prebuilt SPA from web/dist.
  api     Backend only.
  web     Frontend dev server only.
  build   One-shot: install dependencies and build web/dist.

.PARAMETER ApiHost
  Bind address for the API. Default 127.0.0.1.

.PARAMETER ApiPort
  TCP port for the API. Default 18765.

.PARAMETER WebPort
  TCP port for Vite. Default 5173.

.PARAMETER Reload
  Pass --reload to uvicorn so backend code changes trigger a restart.

.PARAMETER NoInstall
  Skip the "first run" pip + npm install steps even if dependencies
  look stale. Useful when iterating on the launcher itself.

.EXAMPLE
  .\scripts\launch.ps1
  .\scripts\launch.ps1 -Mode prod -ApiPort 8080
  .\scripts\launch.ps1 -Mode build
#>

[CmdletBinding()]
param(
  [ValidateSet('dev', 'prod', 'api', 'web', 'build')]
  [string]$Mode = 'dev',
  [string]$ApiHost = '127.0.0.1',
  [int]$ApiPort = 18765,
  [int]$WebPort = 5173,
  [switch]$Reload,
  [switch]$NoInstall
)

$ErrorActionPreference = 'Stop'

# Resolve repo root from the script location so the launcher works no
# matter the user's current directory.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

function Write-Info($msg) { Write-Host "[lorahub] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[lorahub] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[lorahub] $msg" -ForegroundColor Red }

# --- Python --------------------------------------------------------------
function Resolve-Python {
  $venvWin = Join-Path $Root '.venv\Scripts\python.exe'
  if (Test-Path $venvWin) { return $venvWin }
  $venvUnix = Join-Path $Root '.venv/bin/python'
  if (Test-Path $venvUnix) { return $venvUnix }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $cmd = Get-Command py -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw 'No Python interpreter found. Install Python 3.11+ and re-run.'
}

function Test-PythonReady($python) {
  try {
    & $python -c "import lorahub; import fastapi; import uvicorn" 2>$null
    return $LASTEXITCODE -eq 0
  } catch {
    return $false
  }
}

function Install-PythonDeps($python) {
  Write-Info 'Installing Python dependencies (lorahub[api,dev], editable)...'
  & $python -m pip install --upgrade pip
  & $python -m pip install -e ".[api,dev]"
  if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

# --- Node ----------------------------------------------------------------
function Test-NodeReady {
  Test-Path (Join-Path $Root 'web\node_modules\vite')
}

function Install-WebDeps {
  Write-Info 'Installing web dependencies (npm install)...'
  Push-Location (Join-Path $Root 'web')
  try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
  } finally {
    Pop-Location
  }
}

function Build-Web {
  Write-Info 'Building web (vite build)...'
  Push-Location (Join-Path $Root 'web')
  try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'npm run build failed' }
  } finally {
    Pop-Location
  }
}

# --- Port probe ----------------------------------------------------------
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

# --- Service launches ----------------------------------------------------
function Start-Api($python) {
  if (-not (Test-PortFree $ApiPort)) {
    throw "Port $ApiPort is already in use. Pass -ApiPort <other> or stop the process holding it."
  }
  $reloadFlag = if ($Reload) { '--reload' } else { '' }
  $argv = @(
    '-m', 'uvicorn', 'lorahub.api.app:app',
    '--host', $ApiHost,
    '--port', "$ApiPort"
  )
  if ($Reload) { $argv += '--reload' }
  Write-Info "API:  http://${ApiHost}:$ApiPort"
  Start-Process -FilePath $python `
    -ArgumentList $argv `
    -WorkingDirectory $Root `
    -NoNewWindow `
    -PassThru
}

function Start-Web {
  if (-not (Test-PortFree $WebPort)) {
    throw "Port $WebPort is already in use. Pass -WebPort <other> or stop the process holding it."
  }
  $env:LORAHUB_API_TARGET = "http://${ApiHost}:$ApiPort"
  Write-Info "Web:  http://localhost:$WebPort  (proxying /api -> $env:LORAHUB_API_TARGET)"
  $argv = @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$WebPort")
  Start-Process -FilePath 'npm' `
    -ArgumentList $argv `
    -WorkingDirectory (Join-Path $Root 'web') `
    -NoNewWindow `
    -PassThru
}

# --- Main --------------------------------------------------------------
$python = Resolve-Python
Write-Info "Python: $python"

if ($Mode -in 'dev', 'prod', 'api', 'build') {
  if (-not $NoInstall -and -not (Test-PythonReady $python)) {
    Install-PythonDeps $python
  }
}

if ($Mode -in 'dev', 'web', 'build') {
  if (-not $NoInstall -and -not (Test-NodeReady)) {
    Install-WebDeps
  }
}

if ($Mode -eq 'build') {
  Build-Web
  Write-Info 'Build complete. Run -Mode prod to serve the built SPA.'
  return
}

# Make sure the SPA exists before serving in prod mode.
if ($Mode -eq 'prod') {
  $dist = Join-Path $Root 'web\dist\index.html'
  if (-not (Test-Path $dist)) {
    Write-Warn 'web/dist not found — running a build first.'
    if (-not (Test-NodeReady)) { Install-WebDeps }
    Build-Web
  }
}

# Spawn services and keep handles for orderly shutdown on Ctrl+C.
$procs = @()
try {
  if ($Mode -in 'dev', 'prod', 'api') {
    $procs += Start-Api $python
  }
  if ($Mode -in 'dev', 'web') {
    $procs += Start-Web
  }
  Write-Info 'All services launched. Press Ctrl+C to stop.'
  while ($true) {
    Start-Sleep -Seconds 1
    foreach ($p in $procs) {
      if ($p.HasExited) {
        Write-Warn "Service $($p.Id) exited (code=$($p.ExitCode)); shutting down the rest."
        throw 'service exited'
      }
    }
  }
} finally {
  foreach ($p in $procs) {
    if (-not $p.HasExited) {
      try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
  }
}
