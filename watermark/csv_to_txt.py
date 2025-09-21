import csv

# 输入和输出文件路径
csv_file = 'prompts.csv'
txt_file = 'en_prompts_2.txt'

# 打开CSV文件并读取内容
with open(csv_file, mode='r', encoding='utf-8') as infile, \
     open(txt_file, mode='w', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    
    # 遍历每一行，提取 prompt 字段，并写入 txt 文件
    for idx, row in enumerate(reader, start=1):
        prompt = row['prompt'].strip()
        outfile.write(f"{idx}.{prompt}\n")

print(f"已成功将 {csv_file} 转换为 {txt_file}")