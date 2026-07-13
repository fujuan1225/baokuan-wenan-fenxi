#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆款文案分析系统启动脚本
"""

import os
import sys
import webbrowser
import time
from threading import Thread
import subprocess

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'flask',
        'flask-cors',
        'pandas',
        'numpy',
        'scikit-learn',
        'openpyxl',
        'joblib'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺少以下依赖包：{', '.join(missing_packages)}")
        print("正在安装依赖包...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("依赖包安装完成！")

def start_backend():
    """启动后端服务"""
    print("正在启动后端服务...")
    os.system("python3 backend.py &")
    time.sleep(2)  # 等待服务启动

def open_frontend():
    """打开前端界面"""
    frontend_path = os.path.join(os.path.dirname(__file__), '增强版爆款文案分析看板.html')
    if os.path.exists(frontend_path):
        webbrowser.open(f'file://{os.path.abspath(frontend_path)}')
    else:
        print("前端文件不存在，请检查文件路径")

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 爆款文案分析系统启动中...")
    print("=" * 50)

    # 检查依赖
    check_dependencies()

    # 启动后端
    start_backend()

    # 打开前端
    open_frontend()

    print("\n系统启动完成！")
    print("📱 前端界面已在浏览器中打开")
    print("🔧 后端API服务正在运行：http://localhost:5000")
    print("\nAPI文档：")
    print("- 数据概览: http://localhost:5000/api/data-overview")
    print("- 内容预测: http://localhost:5000/api/predict")
    print("- 爆款模板: http://localhost:5000/api/viral-templates")
    print("- 风险分析: http://localhost:5000/api/risk-analysis")
    print("- 导出报告: http://localhost:5000/api/export-report")
    print("- 实时更新: http://localhost:5000/api/real-time-update")

if __name__ == '__main__':
    main()