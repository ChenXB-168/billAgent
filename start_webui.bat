@echo off
chcp 65001 >nul
title 消费管家智能体 - WebUI
cd /d "%~dp0"
echo ============================================================
echo   消费管家智能体 一键启动（WebUI）
echo ------------------------------------------------------------
echo   首次启动约需 10~20 秒，请勿关闭本窗口...
echo   就绪后自动弹出独立应用窗口（无地址栏，可用页面内[退出系统]关闭）
echo   退出：进入应用后点[退出系统]按钮两次，应用窗口与本窗口都会关闭
echo   重复双击：会自动打开在跑系统，不会重复起第二个
echo ============================================================

REM 强制 Python 以 UTF-8 输出（含输出被重定向到文件的场景），避免中文日志乱码
REM 说明：chcp 65001 只影响控制台显示；一旦重定向到文件，Python 会退回系统 locale(GBK)，
REM       故必须显式设置 PYTHONIOENCODING（Python 3.7+ 亦可用 PYTHONUTF8=1）
set PYTHONIOENCODING=utf-8

REM 快速预检：若 7860 已在监听，说明系统正在运行，直接打开独立窗口并结束
netstat -ano | findstr /R /C:":7860 " | findstr /C:"LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo 消费管家智能体 已在运行，自动打开应用窗口：http://127.0.0.1:7860
    if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
        start "" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:7860
    ) else if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" (
        start "" "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:7860
    ) else if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
        start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --app=http://127.0.0.1:7860
    ) else if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
        start "" "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" --app=http://127.0.0.1:7860
    ) else (
        start "" http://127.0.0.1:7860
    )
    exit /b 0
)

if exist "venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -u webUI\app.py
) else (
    python -u webUI\app.py
)
if errorlevel 1 (
    echo.
    echo [启动失败] 详见上方错误信息；若提示端口占用，请先结束残留进程。
    pause
)
