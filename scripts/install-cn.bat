@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub installer — China-region edition with auto mirror selection.
rem
rem Probes a small candidate pool for each download endpoint via
rem PowerShell, picks the fastest reachable one, then forwards to
rem scripts\install.bat. The user does nothing.
rem ----------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "PROBE_PS=%SCRIPT_DIR%install-cn-probe.ps1"

if not exist "%PROBE_PS%" (
  echo [install-cn] missing %PROBE_PS%
  exit /b 1
)

echo [install-cn] selecting fastest mirrors ...
echo.

rem PowerShell prints exactly 5 KEY=VALUE lines (in declared order).
rem We capture them and re-export into this cmd session. Empty values
rem (e.g. GH proxy = direct) are skipped so install.bat falls back to
rem its built-in default for that endpoint.
for /f "usebackq tokens=1,* delims==" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PROBE_PS%"`) do (
  if not "%%a"=="" if not "%%b"=="" set "%%a=%%b"
)

if not defined UV_INDEX_URL (
  echo [install-cn] probe failed; aborting.
  exit /b 1
)

echo.
echo [install-cn] selected mirrors:
if defined LORAHUB_GH_PROXY ( echo   GitHub:  %LORAHUB_GH_PROXY% ) else ( echo   GitHub:  ^(direct^) )
echo   Python:  %UV_PYTHON_INSTALL_MIRROR%
echo   PyPI:    %UV_INDEX_URL%
echo   Node:    %LORAHUB_NODE_MIRROR%
echo   npm:     %NPM_CONFIG_REGISTRY%
echo.

call "%SCRIPT_DIR%install.bat" %*
