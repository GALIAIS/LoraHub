@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub - One-click environment installer (Windows)
rem
rem Installs EVERYTHING into the project directory:
rem   1. uv -> .lorahub\uv\
rem   2. Python 3.12 -> .lorahub\python\
rem   3. Virtual environment -> .venv\
rem   4. Python dependencies (lorahub[api,dev])
rem   5. Node.js portable -> .node\
rem   6. Frontend dependencies (npm install)
rem
rem No pre-existing Python/Node/uv needed. Fully self-contained.
rem After completion, use scripts\run.bat to start.
rem ----------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
pushd "%ROOT%" || (
  echo [ERROR] Cannot cd to project root: %ROOT%
  goto :fail_pause
)

set "TOOLS_DIR=%CD%\.lorahub"
set "UV_DIR=%TOOLS_DIR%\uv"
set "PY_DIR=%TOOLS_DIR%\python"
set "NODE_DIR=%CD%\.node"

echo.
echo ============================================================
echo   LoRaHub Environment Installer
echo ============================================================
echo   Project: %CD%
echo   Tools:   %TOOLS_DIR%
echo.

rem ---- [1/6] Install uv locally -------------------------------------
echo [1/6] Installing uv ...
if exist "%UV_DIR%\uv.exe" (
  echo   OK uv already installed
) else (
  if not exist "%UV_DIR%" mkdir "%UV_DIR%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%UV_DIR%\uv.zip'"
  if errorlevel 1 (
    echo   [ERROR] Failed to download uv.
    goto :fail
  )
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%UV_DIR%\uv.zip' -DestinationPath '%UV_DIR%' -Force"
  if errorlevel 1 (
    echo   [ERROR] Failed to extract uv.
    goto :fail
  )
  rem uv zip extracts into a subdirectory, move files up
  for /d %%d in ("%UV_DIR%\uv-*") do (
    move /Y "%%d\uv.exe" "%UV_DIR%\" >nul 2>&1
    move /Y "%%d\uvx.exe" "%UV_DIR%\" >nul 2>&1
    rd /s /q "%%d" 2>nul
  )
  del "%UV_DIR%\uv.zip" 2>nul
  if not exist "%UV_DIR%\uv.exe" (
    echo   [ERROR] uv.exe not found after extraction.
    goto :fail
  )
  echo   OK uv downloaded
)
set "UV=%UV_DIR%\uv.exe"
set "PATH=%UV_DIR%;%PATH%"
for /f "delims=" %%v in ('"%UV%" --version 2^>nul') do set "UV_VER=%%v"
echo   %UV_VER%
echo.

rem ---- [2/6] Install Python 3.12 locally ----------------------------
echo [2/6] Installing Python 3.12 ...
rem uv lays out two entries per install: a real ``cpython-3.12.<patch>-...``
rem directory and a junction ``cpython-3.12-...`` pointing at it. The
rem junction is uv's stable minor-version alias ??? pinning the venv to
rem the junction means a future ``uv python install 3.12`` (which would
rem repoint the junction to a newer patch) keeps the venv working
rem instead of breaking pyvenv.cfg.
set "PY_EXE="
if exist "%PY_DIR%\cpython-3.12-windows-x86_64-none\python.exe" (
  set "PY_EXE=%PY_DIR%\cpython-3.12-windows-x86_64-none\python.exe"
)
if defined PY_EXE (
  echo   OK Python already installed
) else (
  if not exist "%PY_DIR%" mkdir "%PY_DIR%"
  rem ``--no-bin`` skips uv's per-user shim launcher in
  rem ``%USERPROFILE%\.local\bin``. The project doesn't need it (we
  rem invoke python.exe by full path) and it would otherwise emit a
  rem confusing warning if a prior global ``uv python install`` had
  rem already written a shim there.
  "%UV%" python install 3.12 --install-dir "%PY_DIR%" --no-bin
  if errorlevel 1 (
    echo   [ERROR] Failed to install Python 3.12.
    goto :fail
  )
  if exist "%PY_DIR%\cpython-3.12-windows-x86_64-none\python.exe" (
    set "PY_EXE=%PY_DIR%\cpython-3.12-windows-x86_64-none\python.exe"
  )
  rem Junction missing on older uv builds ??? fall back to the newest
  rem real cpython-3.12.<patch>-... directory.
  if not defined PY_EXE (
    for /d %%d in ("%PY_DIR%\cpython-3.12.*-windows-x86_64-none") do (
      if exist "%%d\python.exe" set "PY_EXE=%%d\python.exe"
    )
  )
  if not defined PY_EXE (
    echo   [ERROR] uv reported success but no python.exe found under
    echo            %PY_DIR%\cpython-3.12*-windows-x86_64-none
    goto :fail
  )
  echo   OK Python 3.12 installed
)
echo   Python: %PY_EXE%
echo.

rem ---- [3/6] Create virtual environment -----------------------------
echo [3/6] Creating virtual environment .venv ...
if exist ".venv\Scripts\python.exe" (
  echo   OK .venv already exists
) else (
  "%UV%" venv .venv --python "%PY_EXE%"
  if errorlevel 1 (
    echo   [ERROR] Failed to create venv.
    goto :fail
  )
  echo   OK .venv created
)
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
echo.

rem ---- [4/6] Install Python dependencies ----------------------------
echo [4/6] Installing Python dependencies ...
"%UV%" pip install -e ".[api,dev]" --python "%VENV_PY%"
if errorlevel 1 (
  echo   [ERROR] pip install failed.
  echo   Try with mirror: "%UV%" pip install -e ".[api,dev]" --python "%VENV_PY%" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  goto :fail
)
echo   OK Python dependencies installed
echo.

rem ---- [5/6] Install Node.js locally --------------------------------
echo [5/6] Installing Node.js ...

rem Always use a project-local portable Node, never the system one.
rem Mixed-version system installs are the single biggest source of
rem "works on my machine" reports for this project, and a portable
rem Node 20 is ~50 MB extracted — small enough that the trade is
rem clearly in our favour.
if exist "%NODE_DIR%\node.exe" (
  if exist "%NODE_DIR%\npm.cmd" (
    set "PATH=%NODE_DIR%;%PATH%"
    for /f "delims=" %%v in ('"%NODE_DIR%\node.exe" --version 2^>nul') do set "NODE_VER=%%v"
    echo   OK Node.js %NODE_VER% (portable, cached)
    goto :node_done
  )
)

echo   Downloading portable Node.js 20 ...
if not exist "%NODE_DIR%" mkdir "%NODE_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.18.1/node-v20.18.1-win-x64.zip' -OutFile '%NODE_DIR%\node.zip'"
if errorlevel 1 (
  echo   [ERROR] Failed to download Node.js.
  goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%NODE_DIR%\node.zip' -DestinationPath '%NODE_DIR%' -Force"
if errorlevel 1 (
  echo   [ERROR] Failed to extract Node.js.
  goto :fail
)
rem Flatten: move contents from node-v20.18.1-win-x64\ up to .node\
for /d %%d in ("%NODE_DIR%\node-v*") do (
  xcopy /E /Y /Q "%%d\*" "%NODE_DIR%\" >nul
  rd /s /q "%%d"
)
del "%NODE_DIR%\node.zip" 2>nul
if not exist "%NODE_DIR%\node.exe" (
  echo   [ERROR] node.exe not found after extraction.
  goto :fail
)
if not exist "%NODE_DIR%\npm.cmd" (
  echo   [ERROR] npm.cmd not found after extraction; archive may
  echo            be corrupted.
  goto :fail
)
set "PATH=%NODE_DIR%;%PATH%"
for /f "delims=" %%v in ('"%NODE_DIR%\node.exe" --version 2^>nul') do set "NODE_VER=%%v"
echo   OK Node.js %NODE_VER% (portable, downloaded)

:node_done
echo.

rem ---- [6/6] Install frontend dependencies --------------------------
echo [6/6] Installing frontend dependencies (web/) ...
if exist "web\node_modules\vite" (
  echo   OK web\node_modules already exists
) else (
  pushd "web" || (
    echo   [ERROR] Cannot enter web directory
    goto :fail
  )
  call npm.cmd install
  set "NPM_RC=!errorlevel!"
  popd
  if not "!NPM_RC!"=="0" (
    echo   [ERROR] npm install failed.
    echo   Try: cd web ^&^& npm install --registry=https://registry.npmmirror.com
    goto :fail
  )
  echo   OK Frontend dependencies installed
)
echo.

echo ============================================================
echo   Installation Complete
echo ============================================================
echo.
echo   All tools installed locally:
echo     uv:      .lorahub\uv\
echo     Python:  .lorahub\python\
echo     Node.js: .node\
echo     venv:    .venv\
echo.
echo   To start LoRaHub:
echo     scripts\run.bat              (default prod: API serves built SPA)
echo     scripts\run.bat dev          (dev mode: API + Vite HMR)
echo.

popd
endlocal
echo Press any key to exit ...
pause >nul
exit /b 0

:fail
popd
:fail_pause
echo.
echo ============================================================
echo   Installation Aborted
echo ============================================================
endlocal
echo Press any key to exit ...
pause >nul
exit /b 1
