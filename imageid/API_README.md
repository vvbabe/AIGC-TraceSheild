# 水印API服务

这个API服务提供了给图片添加水印和从图片中提取水印的功能。

## 目录结构

- `inputs/`: 存放输入图片的目录
- `outputs/`: 存放添加水印后的输出图片的目录

## 启动服务

1. 激活环境：
```
conda activate pic
```

2. 启动API服务（前台运行）：
```
./start_api.sh
```

服务将在 `http://服务器IP:8080` 上运行。

3. 查看日志：
```
tail -f api.log
```

4. 停止服务：
```
./stop_api.sh
```

## API端点

### 1. 添加水印

**URL**: `/add_watermark`

**方法**: `POST`

**参数**:
- `image`: 图片文件 (必须)
- `text`: 要嵌入的文本 (默认为 "GPT")

**示例**:
```bash
curl -X POST -F "image=@inputs/input.png" -F "text=GPT" http://服务器IP:8080/add_watermark -o outputs/output.png
```

### 2. 提取水印

**URL**: `/extract_watermark`

**方法**: `POST`

**参数**:
- `image`: 带水印的图片文件 (必须)

**示例**:
```bash
curl -X POST -F "image=@outputs/output.png" http://服务器IP:8080/extract_watermark
```

**返回**:
```json
{
  "message": "GPT"
}
```

## 使用Python请求示例

### 添加水印
```python
import requests

url = "http://服务器IP:8080/add_watermark"
files = {'image': open('inputs/input.png', 'rb')}
data = {'text': 'GPT'}

response = requests.post(url, files=files, data=data)
with open('outputs/output.png', 'wb') as f:
    f.write(response.content)
```

### 提取水印
```python
import requests
import json

url = "http://服务器IP:8080/extract_watermark"
files = {'image': open('outputs/output.png', 'rb')}

response = requests.post(url, files=files)
result = response.json()
print(f"提取的水印: {result['message']}")
``` 