from watermark_utils import get_path, in_make_responses,get_watermark,load_model
import transformers


def load_chat_history(file_path: str, n: int = 10) -> list[dict]:
    """
    从txt文件加载聊天历史记录，返回最新的n条对话
    
    Args:
        file_path: 聊天历史文本文件路径
        n: 需要获取的最新对话条数，默认为10
        
    Returns:
        格式化后的历史记录列表，格式为：
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 解析对话对
    dialogues = []
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            user_line = lines[i]
            model_line = lines[i+1]
            
            # 检查行是否以User/Model开头
            if user_line.startswith("User:") and model_line.startswith("Model:"):
                user_content = user_line[5:].strip()  # 去掉"User:"
                model_content = model_line[6:].strip()  # 去掉"Model:"
                dialogues.extend([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": model_content}
                ])
    
    # 取最新的n对对话(注意n是指对话对的数量)
    n_pairs = n
    latest_dialogues = dialogues[-(n_pairs*2):] if n_pairs > 0 else []
    
    return latest_dialogues


def chat_give_response(text,history,system_prompt , model_name,flag_watermark,device):
    watermark = get_watermark(model_name)
    format_text = format_prompt(text,model_name,history,system_prompt)

    model = load_model(model_name,device,flag_watermark,watermark)

    response = in_make_responses(format_text,model_name,model,device)

    return response


def get_tokenizer(model_name):
    model_path = get_path(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer



#需要modelname来加载分词器
def format_prompt(text,model_name,history,system_prompt):
    if model_name == "QWEN" or model_name == "DEEPSEEK":
        completed_text = history
        completed_text.insert(0, {"role": "system", "content": system_prompt})
        completed_text.append({"role": "user", "content": text.strip('"')})
    else:
        completed_text = history
        completed_text.append({"role": "user", "content": text.strip('"')})
    tokenizer = get_tokenizer(model_name)
    format_text =  tokenizer.apply_chat_template(
            completed_text, 
            tokenize=False,  # 只格式化，不分词
            add_generation_prompt=True  # 添加模型回复的提示
          )
    
    return format_text
    


