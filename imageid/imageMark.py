from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
from stegastamp import add_watermark, extract_watermark
import logging
import sys
import importlib.util
from pathlib import Path
import subprocess

app = Flask(__name__)
# 启用跨域支持
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type", "Authorization"]}})

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 使用inputs和outputs目录
INPUT_FOLDER = 'inputs'
OUTPUT_FOLDER = 'outputs'

# 确保目录存在
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/add_watermark', methods=['POST'])
def api_add_watermark():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    text = request.form.get('text', 'MARK')
    
    # 记录水印文本
    logger.info(f"添加水印: '{text}'")
    print(f"添加水印: '{text}'")
    
    # 保存上传的图片
    input_path = os.path.join(INPUT_FOLDER, 'input.png')
    image_file.save(input_path)
    
    # 设置输出路径
    output_path = os.path.join(OUTPUT_FOLDER, 'output.png')
    
    # 添加水印
    add_watermark(input_path, text, output_path)
    
    # 返回处理后的图片
    return send_file(output_path, mimetype='image/png')

@app.route('/extract_watermark', methods=['POST'])
def api_extract_watermark():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # 保存上传的图片
    input_path = os.path.join(INPUT_FOLDER, 'watermarked.png')
    image_file.save(input_path)
    
    # 提取水印
    message = extract_watermark(input_path)
    
    # 记录提取的水印
    logger.info(f"提取水印(原始): '{message}'")
    
    # 对提取的水印进行处理，只保留0-9、a-z、A-Z，遇到其他字符则截断
    valid_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    processed_message = ""
    for char in message:
        if char in valid_chars:
            processed_message += char
        else:
            break  # 遇到非法字符就截断
    
    # 记录处理后的水印
    logger.info(f"提取水印(处理后): '{processed_message}'")
    print(f"提取水印: '{processed_message}'")
    
    return jsonify({'message': processed_message})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False) 