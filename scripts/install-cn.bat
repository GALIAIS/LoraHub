@echo off
rem Thin shim: forward to install-cn.ps1 so the actual logic lives in
rem PowerShell (which is robust to LF-only line endings, unlike cmd).
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install-cn.ps1" %*
exit /b %ERRORLEVEL%
