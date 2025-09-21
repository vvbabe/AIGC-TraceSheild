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
from torch.utils.data import Dataset, DataLoader

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

class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]
        
if __name__ == "__main__":
    args = parse_args()

    device_str = args.device if args.device == "cpu" else f"cuda:{args.device}"
    device = torch.device(device_str)




    # 指定你本地数据所在的文件夹路径
    local_dataset_path = "/root/sjt2/synthid-text/data_to_train/data/eli5"
    tokenizer = get_tokenizer(args.model)

    
    data_dir = "/root/sjt2/synthid-text/data_to_train/wikipedia_20231101_en/20231101.en"
    

    # 注意：会自动识别所有 train-*.parquet 文件
    dataset = load_dataset("parquet", data_dir=data_dir, split="train")

    # 只取前 10000 条数据用于示例
    #注：原来是10000，这里改成了1000
    dataset = dataset.select(range(3000))

    # 将 HuggingFace Dataset 转换为 Pandas DataFrame
    df = dataset.to_pandas()

    # 将 DataFrame 转换为 PyTorch Dataset




    # 构建 PyTorch Dataset 和 DataLoader
    texts = df['text'].tolist()
    torch_dataset = TextDataset(texts)
    dataloader = DataLoader(torch_dataset, batch_size=1, shuffle=True)

    # 存储 tokenized 输出
    tokenized_uwm_outputs = []
    batched = []
    padded_length = 2500
    #  开始处理数据
    for i, text in enumerate(tqdm(dataloader)):
        # 解码字符串（如果是 tensor，需先转为 list）
        decoded_text = text[0] if isinstance(text, list) else text[0].item()

        # Tokenize 文本
        inputs = tokenizer(
            decoded_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=padded_length,
            add_special_tokens=False  # 如果你自己添加了 [CLS], [SEP] 等可以设为 False
        ).to(device)

        # 获取 input_ids 并转换为 list
        line = inputs['input_ids'].cpu().numpy()[0].tolist()

        # 截断或填充到统一长度 padded_length
        if len(line) >= padded_length:
            line = line[:padded_length]
        else:
            line += [tokenizer.pad_token_id] * (padded_length - len(line))

        # 添加到当前 batch 中
        batched.append(torch.tensor(line, dtype=torch.long, device=device)[None, :])

        # 当收集到 NEG_BATCH_SIZE 个样本时，合并成一个 batch
        if len(batched) == NEG_BATCH_SIZE:
            tokenized_uwm_outputs.append(torch.cat(batched, dim=0))
            batched = []

        # 控制总处理数量
        if i > NUM_NEGATIVES:
            break
    save_watermark_data_uwm(tokenized_uwm_outputs,args.model + '_datadir')