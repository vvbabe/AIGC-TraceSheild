
from chat_utils import chat_give_response
import torch
#def give_score_bys(text,watermark,device,mytokenizer,bys_path):
# 打开并读取整个文件内容

device = torch.device('cuda:6')

text = chat_give_response('写一篇关于猫头鹰的故事,至少800字',[],'','QWEN',True,device)#没有历史记录，传入[]即可
print(text)
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(text)

