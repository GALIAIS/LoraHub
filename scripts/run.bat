@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub Launcher (Windows)
rem
rem Thin wrapper around the `lorahub` CLI. After running
rem `scripts\install.bat` once, you can invoke
rem `lorahub service start` directly — this script exists for
rem muscle-memory and for setting up the project-local PATH so a
rem fresh shell doesn't need to source the venv first.
rem
rem Usage:
rem   scripts\run.bat              prod mode (foreground uvicorn)
rem   scripts\run.bat dev          dev mode  (uvicorn + Vite HMR)
rem   scripts\run.bat api          alias for prod
rem ----------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
pushd "%ROOT%" || (
  echo [ERROR] Cannot cd to project root: %ROOT%
  goto :fail
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=prod"
set "API_HOST=127.0.0.1"
set "API_PORT=18765"
set "WEB_PORT=6006"

rem ---- Add project-local tools to PATH -------------------------------
if exist ".lorahub\uv\uv.exe" set "PATH=%CD%\.lorahub\uv;!PATH!"
if exist ".node\node.exe"     set "PATH=%CD%\.node;!PATH!"

rem ---- Resolve Python ------------------------------------------------
set "PYTHON="
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=%CD%\.venv\Scripts\python.exe"
) else (
  echo [ERROR] .venv not found. Run scripts\install.bat first.
  goto :fail
)

rem ---- Verify dependencies -------------------------------------------
"%PYTHON%" -c "import lorahub, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python dependencies not installed. Run scripts\install.bat first.
  goto :fail
)

rem ---- Dev mode: vite + uvicorn (two foreground children) -----------
if "%MODE%"=="dev" (
  if not exist ".node\node.exe" (
    echo [ERROR] Portable Node.js missing. Run scripts\install.bat first.
    goto :fail
  )
  echo.
  echo ============================================================
  echo   LoRaHub - dev mode
  echo ============================================================
  echo   API target ^(proxied^):  http://%API_HOST%:%API_PORT%
  echo   Open in browser:       http://localhost:%WEB_PORT%
  echo ============================================================
  echo.
  start "" /B "%PYTHON%" -m uvicorn lorahub.api.app:app --host %API_HOST% --port %API_PORT% --reload
  set "LORAHUB_API_TARGET=http://%API_HOST%:%API_PORT%"
  pushd web
  start "" /B npm.cmd run dev -- --host 127.0.0.1 --port %WEB_PORT%
  popd
  echo.
  echo [lorahub] Services running. Press Ctrl+C to stop.
:loop
  timeout /t 2 /nobreak >nul
  goto :loop
)

rem ---- Prod / api: build SPA if missing, then run via the CLI -------
if not exist "web\dist\index.html" (
  echo [lorahub] Building frontend SPA ...
  "%PYTHON%" -m lorahub manage build
  if errorlevel 1 goto :fail
)

echo [lorahub] starting on http://%API_HOST%:%API_PORT%
"%PYTHON%" -m lorahub service start --host %API_HOST% --port %API_PORT% --foreground
goto :end

:fail
popd 2>nul
endlocal
echo.
echo Press any key to exit ...
pause >nul
exit /b 1

:end
popd 2>nul
endlocal
exit /b 0
