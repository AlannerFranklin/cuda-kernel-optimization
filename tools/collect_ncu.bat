@echo off
REM ============================================================
REM  Collect Nsight Compute data for representative kernels.
REM  MUST be run from an ADMIN terminal (ERR_NVGPUCTRPERM otherwise).
REM  Output: docs\ncu\<kernel>.csv
REM
REM  NOTE: keep this file ASCII-only. Chinese text breaks cmd.exe
REM        parsing under the GBK code page on zh-CN Windows.
REM
REM  Uses a smaller N so each kernel is profiled in a few seconds.
REM ============================================================
setlocal

REM --- require admin ---
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not running as Administrator.
    echo         Right-click Terminal and choose "Run as administrator".
    exit /b 1
)

set "ROOT=%~dp0.."
set "OUT=%ROOT%\docs\ncu"
if not exist "%OUT%" mkdir "%OUT%"

REM Small N: only for counter collection, not for performance numbers.
set "N_SMALL=4194304"

set "METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,dram__bytes.sum"

echo ============================================
echo  ncu collection (9 kernels, a few seconds each)
echo ============================================
echo.

call :profile vector_add         "%ROOT%\build\vector_add\vector_add.exe" %N_SMALL%
call :profile vector_add_strided "%ROOT%\build\vector_add\vector_add.exe" %N_SMALL%
call :profile reduction_v0       "%ROOT%\build\reduction\reduction.exe"  %N_SMALL%
call :profile reduction_v6       "%ROOT%\build\reduction\reduction.exe"  %N_SMALL%
call :profile matmul             "%ROOT%\build\matmul\matmul.exe"        1024
call :profile matmul_reg8        "%ROOT%\build\matmul\matmul.exe"        1024
call :profile histogram_naive    "%ROOT%\build\histogram\histogram.exe"  %N_SMALL%
call :profile histogram_shared   "%ROOT%\build\histogram\histogram.exe"  %N_SMALL%
call :profile softmax            "%ROOT%\build\softmax\softmax.exe"

echo.
echo [OK] Done. CSVs are in docs\ncu\
endlocal
exit /b 0

REM ------------------------------------------------------------
:profile
REM %1 = kernel name, %2 = exe, %3 = optional N
set "KNAME=%~1"
set "EXE=%~2"
echo --- %KNAME% ---
if "%~3"=="" (
    ncu --kernel-name "regex:^%KNAME%$" --launch-count 1 --csv --page raw --metrics %METRICS% "%EXE%" > "%OUT%\%KNAME%.csv"
) else (
    ncu --kernel-name "regex:^%KNAME%$" --launch-count 1 --csv --page raw --metrics %METRICS% "%EXE%" %~3 > "%OUT%\%KNAME%.csv"
)
if errorlevel 1 (
    echo     [WARN] %KNAME% failed, see ncu output above
) else (
    echo     OK -^> docs\ncu\%KNAME%.csv
)
exit /b 0
