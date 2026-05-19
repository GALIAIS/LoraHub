@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub Launcher (Windows)
rem
rem Starts the API backend (uvicorn) and frontend dev server (Vite).
rem Uses project-local tools (.tools\, .node\, .venv\).
rem
rem Usage:
rem   scripts\run.bat              dev mode (API + Vite HMR)
rem   scripts\run.bat prod         production (API + prebuilt SPA)
rem   scripts\run.bat api          API only
rem ----------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
pushd "%ROOT%" || (
  echo [ERROR] Cannot cd to project root: %ROOT%
  goto :fail
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=dev"
set "API_HOST=127.0.0.1"
set "API_PORT=18765"
set "WEB_PORT=6006"

rem ---- Add project-local tools to PATH -------------------------------
if exist ".tools\uv\uv.exe" set "PATH=%CD%\.tools\uv;!PATH!"
if exist ".node\node.exe" set "PATH=%CD%\.node;!PATH!"

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

rem ---- Start API -----------------------------------------------------
if "%MODE%"=="dev" goto :start_api
if "%MODE%"=="prod" goto :start_api
if "%MODE%"=="api" goto :start_api
goto :skip_api

:start_api
echo [lorahub] API:  http://%API_HOST%:%API_PORT%
start "" /B "%PYTHON%" -m uvicorn lorahub.api.app:app --host %API_HOST% --port %API_PORT%
:skip_api

rem ---- Build SPA for prod mode ---------------------------------------
if "%MODE%"=="prod" (
  if not exist "web\dist\index.html" (
    echo [lorahub] Building frontend SPA ...
    pushd web
    call npm.cmd run build
    popd
  )
)

rem ---- Start Web dev server ------------------------------------------
if not "%MODE%"=="dev" goto :skip_web
echo [lorahub] Web:  http://localhost:%WEB_PORT%
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
