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



class ModelName(enum.Enum):
    GPT2 = ('gpt2', '/root/sjt2/pdw/LLM/GPT2',WATER_MARK_CONFIG,'')
    GEMMA_2B = ('google/gemma-2b-it', '/root/sjt2/pdw/LLM/gemma-2b-it',WATER_MARK_GEMMA,'/root/sjt2/synthid-text/data_to_train/bayesian_detector.pkl')
    GEMMA_7B = ('google/gemma-7b-it', '/root/sjt2/pdw/LLM/gemma-7b-it',WATER_MARK_CONFIG,'')
    DEEPSEEK = ('deepseek-ai/deepseek-llm-7b-chat', '/root/sjt2/pdw/LLM/deepseek-7b',WATER_MARK_DEEPSEEK,'/root/sjt2/watermark/bayesian_detectorDEEPSEEK.pkl')  # 示例路径
    QWEN = ('Qwen/Qwen-7B', '/root/sjt2/pdw/LLM/Qwen2.5-7B-Instruct',WATER_MARK_QWEN,'/root/sjt2/synthid-text/data_to_train/20250606/bayesian_detector_qwen.pkl')  
    
    def __init__(self, model_id, local_path,watermark,detector_path):
        self.model_id = model_id
        self.local_path = local_path
        self.watermark = watermark
        self.detector_path = detector_path


# 根据枚举值获取模型路径的函数
#比如输入QWEN这个字符串，输出模型的路径
def get_path(name):
    try:
        return ModelName[name].local_path
    except KeyError:
        valid_names = [m.name for m in ModelName]
        raise ValueError(f"Invalid model name. Valid options: {valid_names}") from None

def get_watermark(name):
    try:
        return ModelName[name].watermark
    except KeyError:
        valid_names = [m.name for m in ModelName]
        raise ValueError(f"Invalid model name. Valid options: {valid_names}") from None
    
def get_detector_path(name):
    try:
        return ModelName[name].detector_path
    except KeyError:
        valid_names = [m.name for m in ModelName]
        raise ValueError(f"Invalid model name. Valid options: {valid_names}") from None

#注：传入的水印是原水印即可
def load_model(
    model_name: ModelName,
    expected_device: torch.device,
    enable_watermarking: bool = False,
    watermark: Optional[Dict[str, Any]] = None,
) -> transformers.PreTrainedModel:
    """
    根据 model_name 加载对应的模型。
    目前仅支持 Gemma-2B，GPT2 和 Gemma-7B 留空供后续实现。
    """
    watermark_config = add_device_to_watermark(watermark,expected_device)
    model_path = get_path(model_name)

    if model_name == 'GEMMA_2B':
        # Gemma-2B 的处理逻辑
        model_cls = (
            SynthID_Gemma
            if enable_watermarking
            else transformers.GemmaForCausalLM
        )
        
        # 先加载配置
        config = transformers.AutoConfig.from_pretrained(model_path)
        
        # 然后创建模型实例
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config,
            **({'watermark_config': watermark_config} if enable_watermarking else {})
        ).to(expected_device)

        if str(model.device) != str(expected_device):
            raise ValueError(f"Model is on {model.device}, but expected {expected_device}.")
        return model

    elif model_name == 'GPT2':
        # GPT2 的处理逻辑
        model_cls = (
            SynthID_GPT
            if enable_watermarking
            else transformers.GPT2LMHeadModel
        )
        
        # 先加载配置
        config = transformers.AutoConfig.from_pretrained(model_path)
        
        # 然后创建模型实例
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config,
            **({'watermark_config': watermark_config} if enable_watermarking else {})
        ).to(expected_device)

        if str(model.device) != str(expected_device):
            raise ValueError(f"Model is on {model.device}, but expected {expected_device}.")
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model
    



    elif model_name == 'GEMMA_7B':
        # 后续实现
        raise NotImplementedError("Gemma-7B 模型加载尚未实现")
    elif model_name == 'DEEPSEEK':
        model_cls = SynthID_DeepSeek if enable_watermarking else transformers.AutoModelForCausalLM
    
    # 确保模型路径正确
        print(f"Loading Deepseek model from: {model_path}")  # 调试用
    
        try:
        # 先加载配置
            config = transformers.AutoConfig.from_pretrained(model_path)
        
            #传入的水印是immutabledict类型的
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                config=config,
                **({'watermark_config': dict(watermark_config)} if enable_watermarking else {})
            ).to(expected_device)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        # 验证模型加载成功
            if model is None:
                raise ValueError("Failed to load Qwen model - returned None")
            
            return model
        
        except Exception as e:
            print(f"Error loading Qwen model: {str(e)}")
            raise

    elif model_name == 'QWEN':
        model_cls = SynthID_Qwen if enable_watermarking else transformers.AutoModelForCausalLM
    
    # 确保模型路径正确
        print(f"Loading Qwen model from: {model_path}")  # 调试用
    
        try:
        # 先加载配置
            config = transformers.AutoConfig.from_pretrained(model_path)
        
            #传入的水印是immutabledict类型的
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                config=config,
                **({'watermark_config': dict(watermark_config)} if enable_watermarking else {})
            ).to(expected_device)
        
        # 验证模型加载成功
            if model is None:
                raise ValueError("Failed to load Qwen model - returned None")
            
            return model
        
        except Exception as e:
            print(f"Error loading Qwen model: {str(e)}")
            raise
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    


def in_make_responses(model_inputs,model_name,model,device):
    
    model_path = get_path(model_name)

    mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    mytokenizer.pad_token = mytokenizer.eos_token
    mytokenizer.padding_side = "left"

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

# 将每个输出解码为字符串，并存储在列表中
    decoded_outputs = []
    for output in outputs:
        decoded_text = mytokenizer.decode(output, skip_special_tokens=True)
        decoded_outputs.append(decoded_text)

# 如果你想合并所有输出（例如，如果是多个生成的序列）
   # output_str = "\n".join(decoded_outputs)

# 或者如果你只关心第一个输出（通常 batch_size=1 时）
    output_str = decoded_outputs[0] if decoded_outputs else ""

# 返回字符串（或进一步处理）
    return output_str



#注：传入的水印必须是含有device的immutabledict类型
def compute_g_values_and_mask(text,device,model_name,watermark):
    """
    给定一段自然文本，计算其 g_values 和 mask。
    """
    model_path = get_path(model_name)

    mytokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    mytokenizer.pad_token = mytokenizer.eos_token
    mytokenizer.padding_side = "left"
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



def in_give_score_bys(text,modelname,watermark,device,bys_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(get_path(modelname))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    g_values,mask = compute_g_values_and_mask(text,device,modelname,watermark)
    with open(bys_path, "rb") as f:
        bayesian_detector = pickle.load(f)
    score = bayesian_detector.detector_module.score(g_values, mask)
    return score

    

