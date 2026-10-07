@echo off
chcp 65001 >nul
title BillAgent 环境安装助手 (CPU版)

:: 检查是否在根目录
if not exist "requirements.txt" (
    echo [错误] 未找到 requirements.txt，请确保此脚本位于 D:\MyCode\billAgent 根目录下！
    pause
    exit
)

echo ==========================================
echo 正在为 BillAgent 配置环境 (CPU + 清华源)
echo 当前目录: %CD%
echo ==========================================

:: 1. 创建/激活虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo [步骤 1] 首次运行，正在创建虚拟环境...
    python -m venv venv
)
echo [步骤 2] 激活虚拟环境...
call venv\Scripts\activate.bat

:: 2. 升级 pip
echo [步骤 3] 升级 pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 3. 安装 CPU 版 PyTorch (关键：防止误装 CUDA 版)
echo [步骤 4] 安装 CPU 版 PyTorch (文件较大，请耐心等待)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: 4. 安装其余依赖
echo [步骤 5] 安装项目依赖库...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ==========================================
echo 安装完成！
echo 建议运行以下命令验证 Torch 是否为 CPU 版:
echo python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
echo ==========================================
pause