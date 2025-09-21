#!/bin/bash


# 在设置了CUDA_VISIBLE_DEVICES后，nvidia-smi 型号 和 Python 检测的 CUDA 设备 会发生变化
# CUDA 0 即CUDA_VISIBLE_DEVICES 设置的那块显卡
#物理 GPU	nvidia-smi 型号	Python 检测的 CUDA 设备	显存使用
#0	A100-SXM4-40GB	❌ 未显示	17MiB
#1	A100-SXM4-40GB	❌ 未显示	17MiB
#2	A100-SXM4-40GB	❌ 未显示	17MiB
#3	GeForce RTX 4090	cuda:	
#4	GeForce RTX 4090	cuda:	
#5	GeForce RTX 4090	cuda:	
#6	GeForce RTX 4090	cuda:

# 
# 默认使用的GPU编号 1
DEFAULT_GPU=1,2

# 解析命令行参数
while getopts "g:h" opt; do
  case $opt in
    g) GPU_ID=$OPTARG ;;
    h) echo "用法: $0 [-g GPU_ID]"
       echo "  -g GPU_ID    指定要使用的GPU编号 (默认: $DEFAULT_GPU)"
       exit 0 ;;
    \?) echo "无效选项: -$OPTARG" >&2
        exit 1 ;;
  esac
done

# 如果未指定GPU，则使用默认值
if [ -z "$GPU_ID" ]; then
  GPU_ID=$DEFAULT_GPU
fi

echo "将使用GPU $GPU_ID 运行水印服务"

# 1. 检查并启动nginx
echo "检查nginx状态..."
if service nginx status > /dev/null 2>&1; then
    echo "Nginx已在运行"
else
    echo "启动Nginx..."
    service nginx start
    if [ $? -eq 0 ]; then
        echo "Nginx启动成功"
    else
        echo "Nginx启动失败，请检查配置"
    fi
fi

# 2. 启动图片水印服务
echo "正在启动图片水印服务..."
echo "是否为图片水印服务设置GPU限制？(y/n，默认y)"
read -r response
response=${response:-y}
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "图片水印服务将限制使用GPU $GPU_ID"
    cd /root/sjt2/imageid
    CUDA_VISIBLE_DEVICES=$GPU_ID bash start_api.sh
else
    echo "图片水印服务将不受GPU限制"
    cd /root/sjt2/imageid
    bash start_api.sh
fi
echo "图片水印服务启动命令已执行"

# 3. 启动文本水印服务
echo "正在启动文本水印服务..."
echo "是否为文本水印服务设置GPU限制？(y/n，默认y)"
read -r response
response=${response:-y}
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "文本水印服务将限制使用GPU $GPU_ID"
    cd /root/sjt2/synthid-text
    CUDA_VISIBLE_DEVICES=$GPU_ID bash start_api.sh
else
    echo "文本水印服务将不受GPU限制"
    cd /root/sjt2/synthid-text
    bash start_api.sh
fi
echo "文本水印服务启动命令已执行"

echo "所有服务启动完成"
