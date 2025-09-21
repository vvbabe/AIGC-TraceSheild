import pickle

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

def new_g_values_and_mask(text,device,model_name,watermark,mytokenizer):
    """
    给定一段自然文本，计算其 g_values 和 mask。
    """

    formal_watermark = add_device_to_watermark(watermark,device)

    inputs = mytokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = inputs['input_ids']


    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **formal_watermark, top_k=TOP_K, temperature=TEMPERATURE
    )

    g_values = logits_processor.compute_g_values(input_ids=input_ids)
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=input_ids,
        eos_token_id=mytokenizer.eos_token_id,
    )[:, formal_watermark['ngram_len'] - 1:]

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=input_ids
    )

    combined_mask = context_repetition_mask * eos_token_mask

    return g_values.cpu().numpy(), combined_mask.cpu().numpy()

def new_give_score_bys(text,modelname,watermark,device,bys_path,tokenizer):

    g_values,mask = new_g_values_and_mask(text,device,modelname,watermark,tokenizer)
    with open(bys_path, "rb") as f:
        bayesian_detector = pickle.load(f)
    score = bayesian_detector.detector_module.score(g_values, mask)
    return score


def get_text_by_index(index, file_path):
    """
    根据序号从文件中获取对应的文本和token数量（改进版）
    
    参数:
        index: 要查找的序号
        file_path: 文件路径
        
    返回:
        元组 (text, count) 或 None（如果找不到）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_number = None
        current_count = None
        data_lines = []
        in_data_section = False
        found = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('number:'):
                # 遇到新记录，先检查是否已找到目标
                if current_number == index:
                    found = True
                    break
                
                # 解析新记录的序号
                try:
                    current_number = int(line.split(':')[1])
                except (IndexError, ValueError):
                    current_number = None
                current_count = None
                in_data_section = False
                data_lines = []
                
            elif line.startswith('count:'):
                try:
                    current_count = int(line.split(':')[1])
                except (IndexError, ValueError):
                    current_count = None
                    
            elif line == 'data:':
                in_data_section = True
                
            elif in_data_section and current_number is not None:
                data_lines.append(line)
        
        # 检查循环结束后是否找到了目标
        if not found and current_number == index:
            found = True
        
        if found and current_count is not None:
            return ('\n'.join(data_lines), current_count)
        else:
            return None
            
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

device = torch.device('cuda:4')
model_name = 'QWEN'
model_path = get_path(model_name)

mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
mytokenizer.pad_token = mytokenizer.eos_token
mytokenizer.padding_side = "left"

watermark = get_watermark(model_name)

dector_path  = get_detector_path(model_name)

file_path = "data_uwm_en.txt"

with open('scores_uwm_en_QWEN.txt', 'a', encoding='utf-8') as f:
    for index in range(1, 501):  # 1到800的循环
        text,count = get_text_by_index(index, file_path)
        if(count<5):
            continue
        score = new_give_score_bys(text,model_name,watermark,device,dector_path,mytokenizer)
        f.write(f"number: {index}   ")
        f.write(f"score: {score}")
        f.write(f"token: {count}\n")
        f.flush()  # 实时刷新到文件