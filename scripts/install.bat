@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub - One-click environment installer (Windows)
rem
rem !! Mirror script: scripts\install.sh (POSIX / Linux / WSL).
rem !! The 6-step contract is documented in scripts\INSTALL_DESIGN.md
rem !! every change here MUST land in install.sh in the same commit.
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
rem
rem Mirror knobs (set via env). Empty -> upstream default. Inside
rem China users typically want the install-cn.bat wrapper which
rem presets every variable below.
rem
rem   LORAHUB_GH_PROXY         GitHub proxy prefix (e.g. https://gh-proxy.org/)
rem   UV_PYTHON_INSTALL_MIRROR python-build-standalone mirror
rem                            (uv reads natively for `uv python install`)
rem   UV_INDEX_URL             PyPI index for `uv pip install`
rem   LORAHUB_NODE_MIRROR      Node binary mirror base
rem                            (default https://nodejs.org/dist)
rem   NPM_CONFIG_REGISTRY      npm registry (npm reads natively)
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
set "NODE_VERSION=20.19.0"
set "NODE_MIN_VERSION=20.19.0"

if not defined LORAHUB_NODE_MIRROR set "LORAHUB_NODE_MIRROR=https://nodejs.org/dist"

echo.
echo ============================================================
echo   LoRaHub Environment Installer
echo ============================================================
echo   Project: %CD%
echo   Tools:   %TOOLS_DIR%
if defined LORAHUB_GH_PROXY         echo   GH proxy:  %LORAHUB_GH_PROXY%
if defined UV_PYTHON_INSTALL_MIRROR  echo   Python:    %UV_PYTHON_INSTALL_MIRROR%
if defined UV_INDEX_URL              echo   PyPI:      %UV_INDEX_URL%
if not "%LORAHUB_NODE_MIRROR%"=="https://nodejs.org/dist" echo   Node:      %LORAHUB_NODE_MIRROR%
if defined NPM_CONFIG_REGISTRY      echo   npm:       %NPM_CONFIG_REGISTRY%
echo.

if defined UV_INDEX_URL if not defined UV_DEFAULT_INDEX set "UV_DEFAULT_INDEX=%UV_INDEX_URL%"

rem ---- [1/6] Install uv locally -------------------------------------
echo [1/6] Installing uv ...
if exist "%UV_DIR%\uv.exe" (
  echo   OK uv already installed
) else (
  if not exist "%UV_DIR%" mkdir "%UV_DIR%"
  set "UV_URL=https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
  if defined LORAHUB_GH_PROXY set "UV_URL=%LORAHUB_GH_PROXY%!UV_URL!"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!UV_URL!' -OutFile '%UV_DIR%\uv.zip'"
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
rem Detect a stale .venv whose pyvenv.cfg `home = ...` points at a
rem Python that's no longer there (typical after we migrated from
rem .tools/python/ to .lorahub/python/, or after the user manually
rem deleted the runtime). If the base interpreter is gone, the venv
rem itself is unusable — uv pip install --python .venv\... fails with
rem "No virtual environment found". Wipe and rebuild from scratch.
set "VENV_VALID="
if exist ".venv\Scripts\python.exe" (
  if exist ".venv\pyvenv.cfg" (
    for /f "usebackq tokens=1,* delims==" %%a in (".venv\pyvenv.cfg") do (
      if /i "%%~a"=="home " (
        set "VENV_HOME=%%~b"
      ) else if /i "%%~a"=="home" (
        set "VENV_HOME=%%~b"
      )
    )
    rem strip leading space from the value (cmd's tokenizer keeps it)
    if defined VENV_HOME (
      for /f "tokens=* delims= " %%h in ("!VENV_HOME!") do set "VENV_HOME=%%h"
      if exist "!VENV_HOME!\python.exe" set "VENV_VALID=1"
    )
  )
)
if defined VENV_VALID (
  echo   OK .venv already exists
) else (
  if exist ".venv" (
    echo   stale .venv detected ^(home=!VENV_HOME!^); rebuilding
    rmdir /s /q ".venv"
  )
  rem ``--seed`` makes uv install pip / setuptools / wheel into the
  rem fresh venv. Without it the venv has no ``pip`` binary, so users
  rem who ``pip install <pkg>`` from the LoraHub in-app terminal hit
  rem the auto-fallback to ``uv pip``. Seeding pip directly is more
  rem intuitive (and lets third-party tools that subprocess
  rem ``pip install`` keep working).
  "%UV%" venv .venv --python "%PY_EXE%" --seed
  if errorlevel 1 (
    echo   [ERROR] Failed to create venv.
    goto :fail
  )
  echo   OK .venv created ^(seeded with pip / setuptools / wheel^)
)
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
echo.

rem ---- [4/6] Install Python dependencies ----------------------------
echo [4/6] Installing Python dependencies ...
set "PY_DEPS_LOG=%CD%\_uv_python_deps.log"
set "PY_DEPS_INDEX_ARGS="
if defined UV_DEFAULT_INDEX set "PY_DEPS_INDEX_ARGS=--index-url '%UV_DEFAULT_INDEX%'"
echo   uv default index: %UV_DEFAULT_INDEX%
echo   running uv pip install -v -e .[api,dev] --python "%VENV_PY%" --link-mode=copy %PY_DEPS_INDEX_ARGS% ^(log: _uv_python_deps.log^)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%UV%' pip install -v -e '.[api,dev]' --python '%VENV_PY%' --link-mode=copy %PY_DEPS_INDEX_ARGS% 2>&1 | Tee-Object -FilePath '%PY_DEPS_LOG%'; exit $LASTEXITCODE"
if errorlevel 1 (
  echo   [ERROR] pip install failed.
  echo   If your network blocks pypi.org, retry with the China-mirror
  echo   wrapper:  scripts\install-cn.bat
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
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if ([version]('!NODE_VER:v=!') -ge [version]('%NODE_MIN_VERSION%')) { exit 0 } else { exit 1 }"
    if not errorlevel 1 (
      echo   OK Node.js !NODE_VER! ^(portable, cached^)
      goto :node_done
    )
    echo   Cached Node.js !NODE_VER! is below required v%NODE_MIN_VERSION%; reinstalling ...
    rmdir /s /q "%NODE_DIR%"
  )
)

echo   Downloading portable Node.js 20 (mirror: %LORAHUB_NODE_MIRROR%) ...
if not exist "%NODE_DIR%" mkdir "%NODE_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%LORAHUB_NODE_MIRROR%/v%NODE_VERSION%/node-v%NODE_VERSION%-win-x64.zip' -OutFile '%NODE_DIR%\node.zip'"
if errorlevel 1 (
  echo   [ERROR] Failed to download Node.js.
  goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%NODE_DIR%\node.zip' -DestinationPath '%NODE_DIR%' -Force"
if errorlevel 1 (
  echo   [ERROR] Failed to extract Node.js.
  goto :fail
)
rem Flatten: move contents from node-v%NODE_VERSION%-win-x64\ up to .node\
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
set "NEEDS_NPM_INSTALL=0"
if not exist "web\node_modules" set "NEEDS_NPM_INSTALL=1"
if not exist "web\node_modules\.package-lock.json" set "NEEDS_NPM_INSTALL=1"
if exist "web\node_modules\.package-lock.json" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-Item 'web\package-lock.json').LastWriteTimeUtc -gt (Get-Item 'web\node_modules\.package-lock.json').LastWriteTimeUtc -or (Get-Item 'web\package.json').LastWriteTimeUtc -gt (Get-Item 'web\node_modules\.package-lock.json').LastWriteTimeUtc) { exit 1 } else { exit 0 }"
  if errorlevel 1 set "NEEDS_NPM_INSTALL=1"
)
if "%NEEDS_NPM_INSTALL%"=="0" (
  pushd "web" >nul && npm.cmd ls --depth=0 >nul 2>nul
  if errorlevel 1 set "NEEDS_NPM_INSTALL=1"
  popd >nul
)
if "%NEEDS_NPM_INSTALL%"=="0" (
  echo   OK web\node_modules already matches package lock
) else (
  pushd "web" || (
    echo   [ERROR] Cannot enter web directory
    goto :fail
  )
  for /f "delims=" %%r in ('npm.cmd config get registry 2^>nul') do set "NPM_REGISTRY_NOW=%%r"
  echo   npm registry: !NPM_REGISTRY_NOW!
  echo   running npm ci ^(verbose log: web\_npm_install.log^) ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%npm-ci-with-log.ps1"
  set "NPM_RC=!errorlevel!"
  popd
  if not "!NPM_RC!"=="0" (
    echo   [ERROR] npm ci failed.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path 'web\_npm_install.log' -Tail 40"
    echo   If your network blocks registry.npmjs.org, retry with the
    echo   China-mirror wrapper:  scripts\install-cn.bat
    goto :fail
  )
  echo   OK Frontend dependencies installed
)
echo.

rem ---- [extra] register the `lorahub` CLI in the user PATH ----------
rem ``.venv\Scripts\lorahub.exe`` exists after the editable install, but
rem we want users to type ``lorahub`` from any cmd window. ``self install``
rem writes ``%LOCALAPPDATA%\lorahub\bin\lorahub.cmd`` and adds that dir to
rem the user PATH via setx. Errors are non-fatal — the venv shim still
rem works, the user just has to invoke it via the full path.
echo [extra] Registering lorahub CLI ...
"%VENV_PY%" -m lorahub manage install
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
echo     lorahub service start        (background daemon - random port)
echo.
echo   Open a new shell so the updated PATH picks up ``lorahub``.
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
