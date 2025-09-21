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

# 生成参数
BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1024
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.99
#注：传入的水印必须是含有device的immutabledict类型
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


def chat_give_response_for_analize(text,model_name,flag_watermark,device,model):
 #   watermark = get_watermark(model_name)
    format_text = format_prompt(text,model_name,[],'')

  
    response = in_make_responses(format_text,model_name,model,device)

    return response

def get_line_from_file(file_path, line_number):
    """
    从指定路径的文本文件中读取第 line_number 行的内容。
    
    参数:
        file_path (str): 文件的路径
        line_number (int): 行号（从1开始）

    返回:
        str: 对应行的文本内容，如果行号无效则返回错误信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            if 1 <= line_number <= len(lines):
                return lines[line_number - 1].strip()
            else:
                return "错误：行号超出文件总行数。"
                
    except FileNotFoundError:
        return "错误：文件未找到，请检查路径是否正确。"
    except Exception as e:
        return f"发生错误：{e}"

warnings.filterwarnings(
    "ignore",
    message="There is a performance drop because we have not yet implemented the batching rule for aten::take"
)

# 屏蔽 torch.load 的 FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.serialization"
)

device = torch.device('cuda:4')
model_name = 'DEEPSEEK'
model_path = get_path(model_name)

mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
mytokenizer.pad_token = mytokenizer.eos_token
mytokenizer.padding_side = "left"

watermark = get_watermark(model_name)

model_wm = load_model(model_name,device,True,watermark)

dector_path  = get_detector_path(model_name)
"""
# 第一次打开文件用 'w' 模式，清空内容
with open('scores_with_text.txt', 'w', encoding='utf-8') as f:
    pass  # 只是为了清空文件
with open('scores.txt', 'w', encoding='utf-8') as f:
    pass  # 只是为了清空文件

# 然后开始循环，每次都用 'a' 模式追加
"""
with open('scores.txt', 'a', encoding='utf-8') as f, \
     open('scores_with_text.txt', 'a', encoding='utf-8') as f2:
    for i in range(325,501):
        input = get_line_from_file('en_prompt.txt',i)
        print(input)
        format_input = format_prompt(input,model_name,[],'')
     
        output, count= new_make_responses(format_input,model_wm,device,mytokenizer)
        if(count<5):
            continue
   
        score = new_give_score_bys(output,model_name,watermark,device,dector_path,mytokenizer)
   
        print('score' + str(score))
        f2.write(f"number: {i}\n")
        f2.write(f"score: {score}\n")
        f2.write(f"text: {output}\n\n")


        f.write(f"number: {i}   ")
        f.write(f"score: {score}")
        f.write(f"token: {count}\n")

        
        f2.flush()  # 实时刷新到文件
        f.flush()  # 实时刷新到文件

