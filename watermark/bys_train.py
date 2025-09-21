import argparse
import gc
import os
from collections.abc import Sequence
import pickle

import numpy as np
from tqdm import tqdm

from watermark_utils import ModelName,load_model,get_watermark
from watermark_config import add_device_to_watermark
from chat_utils import get_tokenizer

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from synthid_text import logits_processing
from synthid_text import detector_bayesian


# 生成参数
BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1024
TEMPERATURE = 0.5
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

def load_watermark_data(save_dir):
    """
    从本地目录加载 wm_outputs 和 tokenized_uwm_outputs。
    
    参数:
        save_dir (str): 加载路径，默认为 "watermark_data"
        
    返回:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: (wm_outputs, tokenized_uwm_outputs)
    """
    # 检查是否存在保存的文件
    wm_path = os.path.join(save_dir, "wm_outputs.pt")
    uwm_path = os.path.join(save_dir, "tokenized_uwm_outputs.pt")

    if not (os.path.exists(wm_path) and os.path.exists(uwm_path)):
        raise FileNotFoundError(f"找不到指定路径中的数据文件，请确认路径 '{save_dir}' 是否正确。")

    # 加载数据
    wm_outputs = torch.load(wm_path)
    tokenized_uwm_outputs = torch.load(uwm_path)

    print(f" 数据已从 {save_dir} 加载成功")
    return wm_outputs, tokenized_uwm_outputs

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
    
    tokenizer = get_tokenizer(args.model)

    watermark = get_watermark(args.model)
    CONFIG = add_device_to_watermark(watermark, device)
    logits_processor = logits_processing.SynthIDLogitsProcessor(
    **CONFIG, top_k=TOP_K, temperature=TEMPERATURE
    )


    padded_length = 2500

    wm_outputs, tokenized_uwm_outputs = load_watermark_data(save_dir=args.model + '_datadir')

    wm_outputs = [t.to(device) for t in wm_outputs]
    tokenized_uwm_outputs = [t.to(device) for t in tokenized_uwm_outputs]

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    print("\nChecking tensors' devices:")
    for i, t in enumerate(wm_outputs):
        print(f"wm_outputs[{i}] device:", t.device)

    for i, t in enumerate(tokenized_uwm_outputs):
        print(f"tokenized_uwm_outputs[{i}] device:", t.device)

    if hasattr(logits_processor, 'parameters'):
        print("\nlogits_processor parameters device:")
        print(next(iter(logits_processor.parameters())).device)
    else:
        print("\nlogits_processor has no parameters.")

    bayesian_detector, test_loss = (
        detector_bayesian.BayesianDetector.train_best_detector(
            tokenized_wm_outputs=wm_outputs,
            tokenized_uwm_outputs=tokenized_uwm_outputs,
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            torch_device=device,
            max_padded_length=MAX_PADDED_LENGTH,
            pos_truncation_length=POS_TRUNCATION_LENGTH,
            neg_truncation_length=NEG_TRUNCATION_LENGTH,
            verbose=True,
            learning_rate=3e-3,
            n_epochs=100,
            l2_weights=np.zeros((1,)),
        )
    )
    print(bayesian_detector.score)
    # 保存
    with open("bayesian_detector"+args.model+".pkl", "wb") as f:
        pickle.dump(bayesian_detector, f)