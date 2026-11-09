@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
rem Local compatibility workaround for duplicate OpenMP in some Anaconda setups.
rem Prefer an isolated Python environment without duplicate runtimes.
set "KMP_DUPLICATE_LIB_OK=TRUE"
python -u 运行实验.py %*
exit /b %errorlevel%
