@echo off
chcp 65001 >nul
title 消费管家智能体 - 全量测试一键回归
cd /d "%~dp0"

REM ============================================================
REM   全量测试一键回归（L1 单测 + 契约检查 + L2 集成 + L3 端到端）
REM ------------------------------------------------------------
REM   设计要点：
REM     ① L1/契约 零外部依赖，任何环境都能跑，是最快的门禁；
REM     ② L2 由 tests/conftest.py 自举 MCP llm_base 子进程（无需手动起服务）；
REM     ③ L3 强制走外部大模型（tests/conftest.py 设 FORCE_EXTERNAL=1 +
REM        DISABLE_LOCAL_LLM=1），**未配 API Key 时必然失败**，故自动跳过并给提示；
REM     ④ 每次 pytest 是独立进程，其 MCP 子进程随 fixture teardown 关闭，
REM        故 L2 与 L3 之间不会端口冲突。
REM ============================================================

REM 强制 Python 以 UTF-8 输出（重定向到文件时也不会乱码）
set PYTHONIOENCODING=utf-8

REM 与生产入口 webUI/app.py::main() 对齐：localhost 调用绕过系统代理
REM （缺失会让 httpx 连 MCP 服务走 proxy_request 而全挂，见设计 07 踩坑记录）
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

if exist "venv\Scripts\python.exe" (
    set "PY=%~dp0venv\Scripts\python.exe"
) else (
    echo [警告] 未找到 venv\Scripts\python.exe，回退使用系统 python
    set "PY=python"
)

set L1=SKIP
set CT=SKIP
set L2=SKIP
set L3=SKIP

echo ============================================================
echo   全量测试一键回归
echo ------------------------------------------------------------
echo   解释器：%PY%
echo   步骤：① L1 单元测试  ② 契约漂移检查  ③ L2 集成  ④ L3 端到端
echo ============================================================
echo.

REM ==================== 1/4 L1 单元测试（零外部依赖） ====================
echo [1/4] L1 单元测试（零外部依赖，约 40s）...
"%PY%" -m pytest tests/unit -q
if errorlevel 1 (set L1=FAIL) else (set L1=PASS)
echo.

REM ==================== 2/4 契约漂移检查（文档 vs 代码） ====================
echo [2/4] 契约漂移检查（设计_从零系列/09 与代码零漂移）...
"%PY%" utils\contract_check.py
if errorlevel 1 (set CT=FAIL) else (set CT=PASS)
echo.

REM ==================== 3/4 L2 集成测试（自举 MCP llm_base） ====================
echo [3/4] L2 集成测试（tests/conftest.py 自动拉起 MCP llm_base）...
"%PY%" -m pytest tests/integration -q
if errorlevel 1 (set L2=FAIL) else (set L2=PASS)
echo.

REM ==================== 4/4 L3 端到端（需外部大模型 API） ====================
echo [4/4] L3 端到端（强制外部模型，耗时较长）...
if "%BILLAGENT_EXT_LLM_API_KEY%"=="" (
    echo [跳过] 未检测到环境变量 BILLAGENT_EXT_LLM_API_KEY。
    echo        e2e 强制走外部模型（tests/conftest.py 设 FORCE_EXTERNAL=1 +
    echo        DISABLE_LOCAL_LLM=1），未配 API 时用例必然失败，故跳过。
    echo        配置方法：复制 .env.example 为 .env 并填第一节的 base_url / api_key。
    echo.
    echo        如需单独跑 e2e，可先配置后执行：
    echo            .\venv\Scripts\python.exe -m pytest tests\e2e -v
    set L3=SKIP
) else (
    "%PY%" -m pytest tests/e2e -q
    if errorlevel 1 (set L3=FAIL) else (set L3=PASS)
)
echo.

REM ==================== 5/5 L3 回答质量评测（可选，耗时长） ====================
set EVAL=SKIP
if /i "%~1"=="--with-eval" (
    echo [5/5] L3 回答质量评测（12 条用例，约 1.5 小时，会消耗 API 额度）...
    if "%BILLAGENT_EXT_LLM_API_KEY%"=="" (
        echo [跳过] 未检测到 BILLAGENT_EXT_LLM_API_KEY，评测强制走外部模型，无法运行。
        set EVAL=SKIP
    ) else (
        "%PY%" evaluation\agent_answer_eval.py
        if errorlevel 1 (set EVAL=FAIL) else (set EVAL=PASS)
    )
) else (
    echo [5/5] L3 回答质量评测：**默认跳过**（12 条用例约 1.5 小时、消耗 API 额度）
    echo        如需一并运行：run_all_tests.bat --with-eval
    set EVAL=SKIP
)
echo.

REM ==================== 汇总 ====================
echo ============================================================
echo   回归结果汇总
echo ------------------------------------------------------------
echo   [1] L1 单元测试      : %L1%
echo   [2] 契约漂移检查     : %CT%
echo   [3] L2 集成测试      : %L2%
echo   [4] L3 端到端        : %L3%   （需外部 API Key）
echo ============================================================

if "%L1%%CT%%L2%"=="PASSPASSPASS" (
    echo   核心门禁（L1 + 契约 + L2）全部通过 ✅
    echo   L3 若为 SKIP：配置 .env 后可再跑一次以覆盖端到端链路。
) else (
    echo   存在失败项，请查看上方输出定位 ⚠️
)
echo.
pause
