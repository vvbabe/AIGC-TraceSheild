
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


def new_make_responses(model_inputs, model, device, mytokenizer):
    inputs = mytokenizer(
        model_inputs,
        return_tensors='pt',
        padding=True,
    ).to(device)

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    seed = random.randint(0, 1000000)  # 随机生成一个 seed
    torch.manual_seed(seed)

    _, inputs_len = inputs['input_ids'].shape

    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_length=inputs_len + OUTPUTS_LEN,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    outputs = outputs[:, inputs_len:]

    # 处理第一个输出
    first_output = outputs[0] if outputs.shape[0] > 0 else []
    decoded_text = mytokenizer.decode(first_output, skip_special_tokens=True)
    token_count = len(first_output)  # 获取token数量

    return decoded_text, token_count  # 返回文本和token数量


def write_text_with_token_count(text, token_count, file_path):
    """
    将文本和token数量追加写入文件
    
    参数:
        text: 要写入的文本内容
        token_count: token数量
        file_path: 文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_entries = f.read().count('number:')
            next_number = existing_entries + 1
    except FileNotFoundError:
        next_number = 1
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"number:{next_number}\n")
        f.write(f"count:{token_count}\n")
        f.write("data:\n")
        f.write(f"{text}\n\n")  # 两个换行作为条目分隔

def get_text_by_index(index, file_path):
    """
    根据序号获取对应的数据和count
    
    参数:
        index: 要查找的序号
        file_path: 文件路径
        
    返回:
        元组 (text, count) 或 None（如果找不到）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    
    entries = content.split('\n\n')
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 3 and lines[0].startswith('number:'):
            current_index = int(lines[0][7:])
            if current_index == index:
                count = int(lines[1][6:])
                text = '\n'.join(lines[3:]) if len(lines) > 3 else ''
                return text, count  # 直接返回元组
    
    return None




def main():
    # 读取prompt.txt中的所有prompt
    with open('en_prompt.txt', 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    device = torch.device('cuda:6')
    model_name = 'QWEN'
    model_path = get_path(model_name)

    mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    mytokenizer.pad_token = mytokenizer.eos_token
    mytokenizer.padding_side = "left"
    watermark = get_watermark(model_name)
    
    model_uwm = load_model(model_name,device,False,watermark)


    # 确保只处理前800个prompt（如果文件中有更多）
    prompts = prompts[258:500]
    

    for i, prompt in enumerate(prompts, 1):
        prompt = prompt.strip()  # 去除首尾空白字符
        if not prompt:  # 跳过空行
            continue
            
        print(f"正在处理第 {i} 个prompt: {prompt[:50]}...")  # 打印前50个字符作为预览
        
        # 获取模型响应
        text,count = new_make_responses(prompt,model_uwm,device,mytokenizer)
        
        # 将响应写入data_uwm.txt
        write_text_with_token_count(text, count, "data_uwm_en.txt")
        
            
        print(f"第 {i} 个prompt处理完成")
    
    print("所有800个prompt处理完成！")

if __name__ == "__main__":
    main()
