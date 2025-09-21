import argparse
import gc
import os
from collections.abc import Sequence

from tqdm import tqdm
from watermark_utils import ModelName,load_model,get_watermark
from watermark_config import add_device_to_watermark
from chat_utils import get_tokenizer

import torch
from datasets import load_dataset

# 生成参数
BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1024
TOP_K = 40
TOP_P = 0.99

# 检测器训练参数
NUM_NEGATIVES = 10000
POS_BATCH_SIZE = 32

NUM_POS_BATCHES = 35
#注：原来是NUM_POS_BATCHES = 313，太多了
NEG_BATCH_SIZE = 32
POS_TRUNCATION_LENGTH = 200
NEG_TRUNCATION_LENGTH = 200
MAX_PADDED_LENGTH = 1000
TEMPERATURE = 0.5

device = torch.device('cuda:1')  


def save_watermark_data_wm(wm_outputs, save_dir):

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 保存带水印的数据
    torch.save(wm_outputs, os.path.join(save_dir, "wm_outputs.pt"))
    print(f"带水印的训练数据已保存至: {save_dir}")

def save_watermark_data_uwm(uwm_outputs, save_dir="watermark_data"):
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    torch.save(uwm_outputs, os.path.join(save_dir, "tokenized_uwm_outputs.pt"))

    print(f"不带水印的训练数据数据已保存至: {save_dir}")





def _process_raw_prompt(prompt: Sequence[str],tokenizer) -> str:

        return tokenizer.apply_chat_template(
            [
            {'role': 'user', 'content': prompt.decode().strip('"')}],
            tokenize=False,
            add_generation_prompt=True,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Run text generation with watermarking detection.")
    parser.add_argument(
        "--model",
        type=str,
        choices=[e.name for e in ModelName],
        required=True,
        help="Model to use (must be one of the ModelName enum names).",
        default='GEMMA_2B'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="1",
        help="Device to use, can be 'cpu' or a GPU index like '0', '1', '2'."
    )
    return parser.parse_args()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()

    device_str = args.device if args.device == "cpu" else f"cuda:{args.device}"
    device = torch.device(device_str)

    watermark = get_watermark(args.model)

    torch.manual_seed(0)
    model = load_model(args.model,device,True,watermark)

    # 指定你本地数据所在的文件夹路径
    local_dataset_path = "/root/sjt2/synthid-text/data_to_train/data/eli5"
    tokenizer = get_tokenizer(args.model)
    # 从本地加载数据集
    eli5_prompts = load_dataset(
        "json",
        data_files={
            "train": "/root/sjt2/synthid-text/data_to_train/data/eli5/train.json",
            "validation": "/root/sjt2/synthid-text/data_to_train/data/eli5/validation.json",
            "test": "/root/sjt2/synthid-text/data_to_train/data/eli5/test.json"
        }
    )
    wm_outputs = []

    for batch_id in tqdm(range(NUM_POS_BATCHES)):
        prompts = eli5_prompts['train']['title'][
            batch_id * POS_BATCH_SIZE:(batch_id + 1) * POS_BATCH_SIZE]
        prompts = [_process_raw_prompt(prompt.encode(),tokenizer) for prompt in prompts]
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
        ).to(device)
        _, inputs_len = inputs['input_ids'].shape

        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=inputs_len + OUTPUTS_LEN,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        )
        wm_outputs.append(outputs[:, inputs_len:])

        del outputs, inputs, prompts

    del model
    gc.collect()
    torch.cuda.empty_cache()
    save_watermark_data_wm(wm_outputs,args.model + '_datadir')