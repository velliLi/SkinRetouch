@echo off
rem Skin Retouch console service - one double-click switch.
rem
rem   double-click            start if stopped, stop if running
rem   right-click > edit      to add flags, e.g. service.py toggle --lan
rem
rem The window is only a reporter: the server itself is detached and keeps
rem running after this window closes.
chcp 65001 >nul 2>nul
title Skin Retouch service switch
setlocal
cd /d "%~dp0"

rem Prefer the interpreter that has OpenCV and torch installed.
set "PY=E:\comfyui\python_embeded\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0service.py" toggle %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" echo Exit code: %CODE%
echo Closing in 8 seconds. Press any key to close now.
timeout /t 8 >nul 2>nul
endlocal
