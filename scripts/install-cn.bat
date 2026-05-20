@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

rem ----------------------------------------------------------------
rem LoRaHub installer — China mirrors preset.
rem
rem Forwards to scripts\install.bat with all download endpoints
rem flipped to in-China mirrors. Every variable below is documented
rem at the top of install.bat; you can mix-and-match by exporting
rem them yourself before invoking install.bat directly.
rem ----------------------------------------------------------------

rem uv release tarball: GitHub via gh-proxy.org
if not defined LORAHUB_GH_PROXY set "LORAHUB_GH_PROXY=https://gh-proxy.org/"

rem python-build-standalone: npmmirror's mirror that uv knows how to use.
if not defined UV_PYTHON_INSTALL_MIRROR set "UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone"

rem PyPI: TUNA (Tsinghua) — the largest and most consistently
rem up-to-date of the in-China PyPI mirrors.
if not defined UV_INDEX_URL set "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

rem Node binary releases: Aliyun's npmmirror also hosts these.
if not defined LORAHUB_NODE_MIRROR set "LORAHUB_NODE_MIRROR=https://npmmirror.com/mirrors/node"

rem npm package registry: same mirror.
if not defined NPM_CONFIG_REGISTRY set "NPM_CONFIG_REGISTRY=https://registry.npmmirror.com"

echo [install-cn] using China mirrors:
echo   GitHub:  %LORAHUB_GH_PROXY%
echo   Python:  %UV_PYTHON_INSTALL_MIRROR%
echo   PyPI:    %UV_INDEX_URL%
echo   Node:    %LORAHUB_NODE_MIRROR%
echo   npm:     %NPM_CONFIG_REGISTRY%
echo.

set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%install.bat" %*
