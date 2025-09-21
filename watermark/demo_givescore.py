from detector_utils import give_watermark_score
import torch

# 有GEMMA_2B,DEEPSEEK和QWEN

device = torch.device('cuda:5')
"""
with open('output.txt', 'r', encoding='utf-8') as file:
    text = file.read()
"""
text = " Respond in 2 sentences. A movie can be memorable due to its compelling storyline, well-developed characters, or powerful emotional impact. Additionally, unique cinematography, iconic music scores, and groundbreaking special effects can also contribute to a film's lasting impression."
score = give_watermark_score(text,'DEEPSEEK',device)
print(score)