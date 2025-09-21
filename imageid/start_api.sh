#!/bin/bash

# 激活conda环境
source /root/sjt2/anaconda3/etc/profile.d/conda.sh
conda activate pic

# 停止之前可能运行的实例
pkill -f "python imageMark.py" || true

# 启动API服务（后台运行）
nohup python imageMark.py > api.log 2>&1 &

# 输出提示信息
echo "图像水印API服务已在后台启动，进程ID: $!"
echo "可以通过 'tail -f api.log' 查看日志" 