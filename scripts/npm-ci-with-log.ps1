$ErrorActionPreference = 'Stop'

$stdoutLog = '_npm_install.log'
$stderrLog = '_npm_install.log.err'

Remove-Item -LiteralPath $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

$process = Start-Process `
    -FilePath 'npm.cmd' `
    -ArgumentList @(
        'ci',
        '--verbose',
        '--no-audit',
        '--no-fund',
        '--fetch-timeout=60000',
        '--fetch-retries=2',
        '--fetch-retry-mintimeout=5000',
        '--fetch-retry-maxtimeout=20000'
    ) `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru `
    -NoNewWindow

while (-not $process.HasExited) {
    Write-Host ("  npm ci still running ... ({0})" -f (Get-Date -Format 'HH:mm:ss'))
    Start-Sleep -Seconds 15
}

if (Test-Path -LiteralPath $stderrLog) {
    Get-Content -LiteralPath $stderrLog | Add-Content -LiteralPath $stdoutLog
    Remove-Item -LiteralPath $stderrLog -Force
}

exit $process.ExitCode
