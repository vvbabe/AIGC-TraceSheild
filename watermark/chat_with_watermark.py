import argparse
from watermark_utils import ModelName
from chat_utils import load_chat_history , chat_give_response


import torch


# ========== 命令行参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Run text generation with watermarking detection.")
    parser.add_argument(
        "--model",
        type=str,
        choices=[e.name for e in ModelName],
        required=False,
        help="Model to use (must be one of the ModelName enum names).",
        default='GEMMA_2B'
    )
    parser.add_argument(
        "--history_file",
        type=str,
        required=False,
        help="history txt",
        default='AI_history.txt'
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="output txt",
        default='output.txt'
    )   
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="prompt",
        default=''
    )
    parser.add_argument(
        "--device",
        type=str,
        default="2",
        help="Device to use, can be 'cpu' or a GPU index like '0', '1', '2'."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant",
        help="system_prompt"
    )
    parser.add_argument(
        "--history_num",#读取历史记录的条数
        type=int,
        default="10",
        help="the num of history to read"
    )
    parser.add_argument(
        "--enable_watermark",#如果添加了--enable_watermark，就是有水印
        action="store_true",
        required=False,
        help="Whether to enable watermark during generation."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    try:
        history = load_chat_history(args.history_file, args.history_num)
    except FileNotFoundError:
        history = []
        
    device_str = args.device if args.device == "cpu" else f"cuda:{args.device}"
    device = torch.device(device_str)

    output = chat_give_response(args.prompt,history,args.system_prompt,args.model,args.enable_watermark,device)

    print("output: " + output)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(output)

        
    with open(args.history_file, "a", encoding="utf-8") as f:  # 'a' 表示追加模式
            f.write(f"User: {args.prompt}\n")
            f.write(f"Model: {output}\n")
            f.write('\n')



    
