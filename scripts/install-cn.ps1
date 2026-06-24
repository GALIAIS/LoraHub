# LoRaHub installer — China-region edition with auto mirror selection.
#
# This is the actual logic. install-cn.bat is a one-line cmd shim
# that calls into here so a .bat file with broken line endings (a
# recurring trap when editors save LF-only) doesn't take the whole
# installer down.

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProbeScript = Join-Path $ScriptDir 'install-cn-probe.ps1'
$InstallBat = Join-Path $ScriptDir 'install.bat'

if (-not (Test-Path $ProbeScript)) {
    Write-Host "[install-cn] missing $ProbeScript" -ForegroundColor Red
    exit 1
}

Write-Host "[install-cn] selecting fastest mirrors ..."
Write-Host ""

# Capture KEY=VALUE lines from the probe.
$probeOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $ProbeScript
foreach ($line in $probeOutput) {
    if ($line -match '^([A-Z_]+)=(.+)$') {
        $name = $matches[1]
        $value = $matches[2]
        # Skip empty values so install.bat falls back to its built-in
        # default (e.g. an empty GH_PROXY means "direct").
        if ($value -ne '') {
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

if (-not $env:UV_INDEX_URL) {
    Write-Host "[install-cn] probe failed; aborting." -ForegroundColor Red
    exit 1
}
$env:UV_DEFAULT_INDEX = $env:UV_INDEX_URL

Write-Host ""
Write-Host "[install-cn] selected mirrors:"
if ($env:LORAHUB_GH_PROXY) {
    Write-Host "  GitHub:  $env:LORAHUB_GH_PROXY"
} else {
    Write-Host "  GitHub:  (direct)"
}
Write-Host "  Python:  $env:UV_PYTHON_INSTALL_MIRROR"
Write-Host "  PyPI:    $env:UV_INDEX_URL"
Write-Host "  Node:    $env:LORAHUB_NODE_MIRROR"
Write-Host "  npm:     $env:NPM_CONFIG_REGISTRY"
Write-Host "  PyTorch: $env:LORAHUB_TORCH_INDEX_URL"
Write-Host ""

# Forward any extra args the user passed verbatim to install.bat.
& cmd.exe /c $InstallBat $args
exit $LASTEXITCODE
