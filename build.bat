@echo off
REM ============================================================
REM  Build every kernel into build\
REM
REM  Usage:  build.bat          build all
REM          build.bat clean    remove build\
REM
REM  NOTE: keep this file ASCII-only. Chinese text breaks cmd.exe
REM        parsing under the GBK code page on zh-CN Windows.
REM
REM  The compiler is auto-detected via vswhere, so this works on
REM  machines with either VS 2022 Build Tools or VS 2026.
REM ============================================================
setlocal

if /i "%~1"=="clean" (
    if exist "%~dp0build" rmdir /s /q "%~dp0build"
    echo [OK] Cleaned %~dp0build
    exit /b 0
)

REM --- locate Visual Studio / Build Tools ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VSDIR=E:\BuildTools"
if not defined VSDIR if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
)
if not defined VSDIR (
    echo [ERROR] MSVC not found. Install Visual Studio 2022 Build Tools:
    echo         winget install Microsoft.VisualStudio.2022.BuildTools
    exit /b 1
)
echo [INFO] Using compiler at: %VSDIR%

set "CMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to load MSVC environment.
    exit /b 1
)

"%CMAKE%" -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%~dp0build"
if errorlevel 1 exit /b 1

echo.
echo [OK] Build succeeded: %~dp0build\
echo      Run any of: reduction.exe, matmul.exe, ...
endlocal
