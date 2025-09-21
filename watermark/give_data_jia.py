import re

def parse_txt_file(file_path):
    """解析文本文件，提取 score 和 token 数据"""
    result = []
    score_pattern = re.compile(r'score: \[(.*?)\]token: (\d+)')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = score_pattern.search(line.strip())
            if match:
                score = float(match.group(1))
                token = int(match.group(2))
                result.append([score, token])
    return result

def calculate_tpr(data, k, a, b):
    """
    计算在 token ∈ [a, b] 范围内，score ≥ k 的真阳性率（TPR），并返回统计信息
    
    Args:
        data (list): 数据列表，格式如 [[score, token], ...]
        k (float): 判断水印的 score 阈值
        a (int): token 下界（最小 token 值）
        b (int): token 上界（最大 token 值）
        
    Returns:
        dict: 包含 TPR、样本数量等统计信息
    """
    # 1. 筛选 token 在 [a, b] 范围内的数据
    filtered_data = [[score, token] for score, token in data if a <= token <= b]
    total_samples = len(filtered_data)
    
    if not filtered_data:
        return {
            "tpr": 0.0,
            "total_samples": 0,
            "true_positives": 0,
            "threshold": k,
            "token_range": [a, b]
        }
    
    # 2. 计算 score ≥ k 的数量
    true_positives = sum(1 for score, _ in filtered_data if score >= k)
    
    # 3. 计算 TPR
    tpr = true_positives / total_samples
    
    return {
        "tpr": tpr,
        "total_samples": total_samples,
        "true_positives": true_positives,
        "threshold": k,
        "token_range": [a, b]
    }

# 解析数据并合并
list_a = parse_txt_file("假阳性数据文件/QWEN_en.txt")


# 计算 TPR 并打印详细结果（a=0 是下界，b=100 是上界）
stats = calculate_tpr(list_a, k=0.5, a=0, b=100)
print("统计结果:")
print(f"- Token 范围: {stats['token_range'][0]} ~ {stats['token_range'][1]}")
print(f"- 总样本数: {stats['total_samples']}")
print(f"- 假阳性数 (score ≥ {stats['threshold']}): {stats['true_positives']}")
print(f"- 假阳性率 (TPR): {stats['tpr']:.4f}")