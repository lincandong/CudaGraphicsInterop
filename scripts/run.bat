@echo off

set tutorial=%1
if "%~1"=="" goto :error
set proj_name=%tutorial%

cmake --build ./build --config Release --target %proj_name%
if not %ERRORLEVEL% == 0 goto :error

build\%tutorial%\Release\%proj_name%.exe %2 %3

if not %ERRORLEVEL% == 0 goto :error

goto :endofscript

:error
echo =========================
echo There was an error!
echo =========================

:endofscript
