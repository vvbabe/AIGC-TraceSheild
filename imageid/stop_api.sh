#!/bin/bash

# 查找运行的Flask进程
PID=$(ps -ef | grep "python imageMark.py" | grep -v grep | awk '{print $2}')
PID_OLD=$(ps -ef | grep "python imageMar.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ] && [ -z "$PID_OLD" ]; then
  echo "没有找到运行中的图像水印API服务"
else
  if [ ! -z "$PID" ]; then
    echo "停止图像水印API服务，进程ID: $PID"
    kill $PID
  fi
  
  if [ ! -z "$PID_OLD" ]; then
    echo "停止旧版API服务，进程ID: $PID_OLD"
    kill $PID_OLD
  fi
  
  echo "图像水印API服务已停止"
fi 