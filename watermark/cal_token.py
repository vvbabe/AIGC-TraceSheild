
from chat_utils import chat_give_response
import torch
from chat_utils import format_prompt
from watermark_utils import compute_g_values_and_mask, get_path, in_make_responses,get_watermark,load_model, get_detector_path


from detector_utils import give_watermark_score
import torch

import transformers
import torch
import warnings

import enum
import gc
import pickle
import random
import transformers
import torch
from typing import Any, Optional, Union,Dict

from synthid_text import logits_processing
from transformers import AutoTokenizer

from watermark_config import SynthID_Qwen,SynthID_Gemma,SynthID_GPT,WATER_MARK_QWEN,WATER_MARK_GEMMA,WATER_MARK_CONFIG,SynthID_DeepSeek,WATER_MARK_DEEPSEEK
from watermark_config import add_device_to_watermark

BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1024
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.99
from tqdm import tqdm  # 用于进度条显示

def calculate_token_counts(input_file, output_file):
    """
    读取输入文件，计算每段文本的 token 数量，并写入输出文件
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.read().split('\n\n')  # 按空行分割每段文本
        
        for segment in tqdm(lines, desc="Processing segments"):
            if not segment.strip():
                continue  # 跳过空行
                
            # 解析每段文本
            lines_in_segment = segment.split('\n')
            number_line = lines_in_segment[0]
            count_line = lines_in_segment[1]
            data_line = '\n'.join(lines_in_segment[2:])  # 合并多行data内容
            
            # 提取data部分的文本（去掉"data:"前缀）
            data_text = data_line.replace('data:', '').strip()
            
            # 计算token数量
            tokens = mytokenizer.encode(data_text, add_special_tokens=False)
            token_count = len(tokens)
            
            # 更新count行
            updated_count_line = f"count:{token_count}"
            
            # 重新组合并写入输出文件
            updated_segment = f"{number_line}\n{updated_count_line}\ndata:\n{data_text}\n"
            f_out.write(updated_segment + '\n')

device = torch.device('cuda:6')
model_name = 'QWEN'
model_path = get_path(model_name)

mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
mytokenizer.pad_token = mytokenizer.eos_token
mytokenizer.padding_side = "left"
watermark = get_watermark(model_name)



# 使用示例
input_file = "test.txt"  # 替换为你的输入文件路径
output_file = "test2.txt"  # 输出文件路径
calculate_token_counts(input_file, output_file)