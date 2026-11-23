@echo off
REM ============================================================
REM  Collect Nsight Compute data for representative kernels.
REM  MUST be run from an ADMIN terminal (ERR_NVGPUCTRPERM otherwise).
REM  Output: docs\ncu\<name>.csv
REM ============================================================
setlocal

REM --- require admin ---
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not running as Administrator.
    echo         Nsight Compute needs admin rights to read GPU counters.
    echo         Right-click Terminal / cmd -^> "Run as administrator".
    exit /b 1
)

set "ROOT=%~dp0.."
set "OUT=%ROOT%\docs\ncu"
if not exist "%OUT%" mkdir "%OUT%"

set "METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,dram__bytes.sum"

call :profile vector_add     vector_add           "%ROOT%\build\vector_add\vector_add.exe"
call :profile vector_add     vector_add_strided   "%ROOT%\build\vector_add\vector_add.exe"
call :profile reduction      reduction_v0         "%ROOT%\build\reduction\reduction.exe"
call :profile reduction      reduction_v6         "%ROOT%\build\reduction\reduction.exe"
call :profile matmul         matmul               "%ROOT%\build\matmul\matmul.exe"
call :profile matmul         matmul_reg8          "%ROOT%\build\matmul\matmul.exe"
call :profile histogram      histogram_naive      "%ROOT%\build\histogram\histogram.exe"
call :profile histogram      histogram_shared     "%ROOT%\build\histogram\histogram.exe"
call :profile softmax        softmax              "%ROOT%\build\softmax\softmax.exe"

echo.
echo [OK] Done. CSVs are in docs\ncu\
endlocal
exit /b 0

REM ------------------------------------------------------------
:profile
REM %1 = tag (file name), %2 = kernel name, %3 = exe
echo === %2 ===
ncu --kernel-name "regex:^%2$" --launch-count 1 --csv --page raw --metrics %METRICS% "%~3" > "%OUT%\%2.csv" 2>nul
if errorlevel 1 (
    echo [WARN] %2 failed ^(check the exe path or the kernel name^)
) else (
    echo       saved to docs\ncu\%2.csv
)
exit /b 0
