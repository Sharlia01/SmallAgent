# 多阶段构建：开发环境
FROM node:18-alpine AS development

# 设置工作目录
WORKDIR /app

# 配置国内镜像源
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories

# 安装必要的构建工具
RUN apk add --no-cache python3 make g++

# 复制 package.json 和 package-lock.json
COPY package*.json ./

# 使用国内镜像源安装依赖
RUN npm config set registry https://registry.npmmirror.com/ && \
    npm install

# 复制源代码
COPY . .

# 暴露开发服务器端口
EXPOSE 5181

# 启动开发服务器
CMD ["npm", "run", "dev"]

# 生产环境构建阶段
FROM node:18-alpine AS builder

WORKDIR /app

# 配置国内镜像源
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories

# 安装构建工具
RUN apk add --no-cache python3 make g++

# 复制依赖文件
COPY package*.json ./

# 安装依赖
RUN npm config set registry https://registry.npmmirror.com/ && \
    npm install

# 复制源代码
COPY . .

# 构建生产版本
RUN npm run build

# 生产环境运行阶段
FROM nginx:alpine AS production

# 复制自定义 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 从构建阶段复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 暴露 HTTP 端口
EXPOSE 80

# 启动 nginx
CMD ["nginx", "-g", "daemon off;"]