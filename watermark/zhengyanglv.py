# 导入必要的库
import re

# 初始化统计变量
tp = 0  # 真阳性
fn = 0  # 假阴性（因为全是正类）

# 尝试读取文件
file_path = 'scores_qwen.txt'

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"❌ 文件未找到，请确认路径是否正确：{file_path}")
    exit()

# 遍历每一行
score_found = False  # 标记是否找到了至少一个 score

for line in lines:
    # 使用正则提取 score 数值
    match = re.search(r'score:\s*\[(.*?)\]', line.strip())
    if match:
        score_found = True
        try:
            score = float(match.group(1))
            if score > 0.5:
                tp += 1
            else:
                fn += 1
        except ValueError:
            print(f"⚠️ 无法转换分数值: {match.group(1)}")
            continue

# 检查是否有 score 被提取到
if not score_found:
    print("❌ 未在文件中找到任何有效的 score 数据，请检查文件格式！")
    exit()

# 计算 TPR
try:
    tpr = tp / (tp + fn)
except ZeroDivisionError:
    print("❌ TP + FN = 0，可能所有样本都没有被正确解析。")
    exit()

# 输出结果
print(f"真阳性数 (TP): {tp}")
print(f"假阴性数 (FN): {fn}")
print(f"真阳性率 (TPR): {tpr:.4f}")