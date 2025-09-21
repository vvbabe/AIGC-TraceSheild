#!/bin/bash

# 1. 停止文本水印服务
echo "正在停止文本水印服务..."
cd /root/sjt2/synthid-text
if [ -f "stop_api.sh" ]; then
    bash stop_api.sh
    echo "文本水印服务停止命令已执行"
else
    echo "尝试查找并终止文本水印服务进程..."
    # 假设服务使用Python启动，可能需要根据实际情况调整
    pkill -f "python.*synthid-text"
    if [ $? -eq 0 ]; then
        echo "文本水印服务已停止"
    else
        echo "未找到运行中的文本水印服务进程"
    fi
fi

# 2. 停止图片水印服务
echo "正在停止图片水印服务..."
cd /root/sjt2/imageid
if [ -f "stop_api.sh" ]; then
    bash stop_api.sh
    echo "图片水印服务停止命令已执行"
else
    echo "尝试查找并终止图片水印服务进程..."
    # 假设服务使用Python启动，可能需要根据实际情况调整
    pkill -f "python.*imageid"
    if [ $? -eq 0 ]; then
        echo "图片水印服务已停止"
    else
        echo "未找到运行中的图片水印服务进程"
    fi
fi

# 3. 停止nginx（可选，取决于是否有其他服务依赖nginx）
echo "是否要停止Nginx服务？(y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "停止Nginx..."
    service nginx stop
    if [ $? -eq 0 ]; then
        echo "Nginx已停止"
    else
        echo "Nginx停止失败，请检查"
    fi
else
    echo "保持Nginx运行"
fi

echo "所有服务停止完成" 