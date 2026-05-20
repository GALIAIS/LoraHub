@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub Launcher (Windows)
rem
rem Starts the API backend (uvicorn). In prod mode the API also
rem serves the prebuilt SPA from web\dist; in dev mode a separate
rem Vite HMR server is started on its own port.
rem
rem Usage:
rem   scripts\run.bat              prod mode (default — API serves SPA)
rem   scripts\run.bat dev          dev mode (API + Vite HMR)
rem   scripts\run.bat api          API only (no SPA build, no Vite)
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
rem Hard requirement: every binary the launcher touches comes from the
rem project tree. We never fall back to system installs — that's the
rem whole point of the install.bat layout.
if exist ".lorahub\uv\uv.exe" set "PATH=%CD%\.lorahub\uv;!PATH!"
if exist ".node\node.exe" (
  set "PATH=%CD%\.node;!PATH!"
) else (
  echo [ERROR] Portable Node.js not found at .node\node.exe.
  echo          Run scripts\install.bat first.
  goto :fail
)

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

echo.
echo ============================================================
echo   LoRaHub - %MODE% mode
echo ============================================================
echo.

rem ---- Build SPA for prod mode (must run before API start so the
rem      static mount in lorahub.api.app picks up web\dist) ----------
if "%MODE%"=="prod" (
  if not exist "web\dist\index.html" (
    echo [lorahub] Building frontend SPA ...
    pushd web
    call npm.cmd run build
    popd
    if not exist "web\dist\index.html" (
      echo [ERROR] Frontend build failed — web\dist\index.html missing.
      goto :fail
    )
  )
)

rem ---- Start API -----------------------------------------------------
if "%MODE%"=="dev" goto :start_api
if "%MODE%"=="prod" goto :start_api
if "%MODE%"=="api" goto :start_api
goto :skip_api

:start_api
if "%MODE%"=="prod" (
  echo [lorahub] Open: http://%API_HOST%:%API_PORT%
) else if "%MODE%"=="api" (
  echo [lorahub] API:  http://%API_HOST%:%API_PORT%
) else (
  rem dev mode — make it obvious which URL is the one to open in the
  rem browser. The API port is only the proxy target; opening it
  rem directly serves the previously-built (or stale) production SPA
  rem and skips Vite HMR + dev-only tooling like the React Query
  rem devtools floating button.
  echo [lorahub] API target ^(proxied^):  http://%API_HOST%:%API_PORT%
)
start "" /B "%PYTHON%" -m uvicorn lorahub.api.app:app --host %API_HOST% --port %API_PORT%
:skip_api

rem ---- Start Web dev server (dev mode only) -------------------------
if not "%MODE%"=="dev" goto :skip_web
echo.
echo [lorahub] ============================================================
echo [lorahub]   Open in browser:  http://localhost:%WEB_PORT%
echo [lorahub]   ^(the API at :%API_PORT% is the proxy target, not the UI^)
echo [lorahub] ============================================================
echo.
set "LORAHUB_API_TARGET=http://%API_HOST%:%API_PORT%"
pushd web
start "" /B npm.cmd run dev -- --host 127.0.0.1 --port %WEB_PORT%
popd
:skip_web

echo.
echo [lorahub] Services running. Press Ctrl+C to stop.
echo.

rem ---- Keep alive until Ctrl+C ---------------------------------------
:loop
timeout /t 2 /nobreak >nul
goto :loop

:fail
popd 2>nul
endlocal
echo.
echo Press any key to exit ...
pause >nul
exit /b 1
