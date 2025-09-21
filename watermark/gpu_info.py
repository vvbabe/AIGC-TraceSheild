import torch
import subprocess

def get_gpu_info():
    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count()
    
    print("PyTorch 检测到的 GPU 信息：")
    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        
        print(f"cuda:{i} - {name}")
        print(f"  Compute Capability: {capability[0]}.{capability[1]}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        
        # 使用 nvidia-smi 查询实时内存和温度等信息
        try:
            smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.free,memory.total',
                 '--format=csv',
                 '-i', str(i)]
            ).decode('utf-8').splitlines()

            if len(smi_output) > 1:
                data = smi_output[1].split(',')
                index = data[0].strip()
                temp = data[1].strip()
                util = data[2].strip()
                mem_used = data[3].strip()
                mem_free = data[4].strip()
                mem_total = data[5].strip()

                print(f"  Temperature: {temp}°C")
                print(f"  GPU Utilization: {util}%")
                print(f"  Memory Used: {mem_used}")
                print(f"  Memory Free: {mem_free}")
                print(f"  Memory Total: {mem_total}")
        except Exception as e:
            print(f"  [无法通过 nvidia-smi 获取详细信息: {e}]")

        print("-" * 60)

if __name__ == '__main__':
    get_gpu_info()