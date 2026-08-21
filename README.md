# 后端启动

cd backend


## 步骤1：修改.env文件中 DASHSCOPE_API_KEY

## 启动后端服务
docker compose up -d --build

查看后端日志：docker logs -f swxy_api


# 前端启动

cd frontend

## 删除旧文件
rm -rf node_modules package-lock.json

## 重新安装依赖
npm install

## 启动开发服务器
npm run dev

# 访问服务
http://localhost:5181/





