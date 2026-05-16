@echo off
rem LoraHub launcher — Windows .bat shim that hands off to launch.ps1.
rem Lets users double-click in Explorer or run "scripts\launch.bat" from cmd.
rem Forwards any arguments verbatim, so:
rem   scripts\launch.bat -Mode prod -ApiPort 8080
rem   scripts\launch.bat -Mode build
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%launch.ps1" %*
exit /b %ERRORLEVEL%
