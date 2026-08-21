#!/bin/bash

# 前端项目启动脚本

echo "====================================="
echo "  深微学院前端项目 Docker 启动脚本"
echo "====================================="
echo ""
echo "请选择启动模式："
echo "1. 开发模式（热重载，端口 5181）"
echo "2. 生产模式（Nginx，端口 80）"
echo ""

read -p "请输入选项 (1 或 2): " choice

case $choice in
    1)
        echo ""
        echo "正在启动开发环境..."
        docker-compose --profile dev up -d
        echo ""
        echo "✅ 开发环境已启动！"
        echo "🌐 访问地址: http://localhost:5181"
        echo "📝 日志查看: docker-compose logs -f frontend-dev"
        echo "🛑 停止服务: docker-compose --profile dev down"
        ;;
    2)
        echo ""
        echo "正在构建并启动生产环境..."
        docker-compose --profile prod up -d --build
        echo ""
        echo "✅ 生产环境已启动！"
        echo "🌐 访问地址: http://localhost:80"
        echo "📝 日志查看: docker-compose logs -f frontend-prod"
        echo "🛑 停止服务: docker-compose --profile prod down"
        ;;
    *)
        echo "❌ 无效选项，请输入 1 或 2"
        exit 1
        ;;
esac

echo ""
echo "🐳 查看所有容器: docker ps -a"
echo "🔍 查看网络: docker network ls"