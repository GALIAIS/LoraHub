@echo off
chcp 65001 >nul 2>&1
setlocal

rem LoRaHub SSH tunnel (Windows)
rem Usage:
rem   scripts\tunnel.bat <user@host> [ssh_port] [local_port] [remote_port]
rem Example:
rem   scripts\tunnel.bat cwadmin@113.108.63.33 13122 18080 18765

set "TARGET=%~1"
set "SSH_PORT=%~2"
set "LOCAL_PORT=%~3"
set "REMOTE_PORT=%~4"

if "%TARGET%"=="" (
  echo Usage: scripts\tunnel.bat ^<user@host^> [ssh_port] [local_port] [remote_port]
  echo Example: scripts\tunnel.bat cwadmin@113.108.63.33 13122 18080 18765
  exit /b 2
)
if "%SSH_PORT%"=="" set "SSH_PORT=22"
if "%LOCAL_PORT%"=="" set "LOCAL_PORT=18080"
if "%REMOTE_PORT%"=="" set "REMOTE_PORT=18765"

where ssh.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ssh.exe not found. Install OpenSSH Client in Windows Optional Features.
  exit /b 1
)

echo.
echo ============================================================
echo   LoRaHub SSH Tunnel
echo ============================================================
echo   SSH:        %TARGET%:%SSH_PORT%
echo   Browser:    http://127.0.0.1:%LOCAL_PORT%/
echo   Forward:    127.0.0.1:%LOCAL_PORT% -^> 127.0.0.1:%REMOTE_PORT%
echo ============================================================
echo Keep this window open while using LoRaHub. Press Ctrl+C to stop.
echo.

ssh.exe -p "%SSH_PORT%" -N -L "127.0.0.1:%LOCAL_PORT%:127.0.0.1:%REMOTE_PORT%" "%TARGET%"
exit /b %ERRORLEVEL%
