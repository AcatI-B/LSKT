import subprocess
import time
import random

cho = 1
dataset=[ 'assist17', 'assist09', 'assist12', 'algebra05','statics']
choice = dataset[cho]   #选择数据集
mode='stage1'   #选择阶段1还是阶段2
# maxat_list = [ 1326, 1184, 4000, 4000]
maxat_list = [ 9999, 84076, 4000, 4000, 4000]
maxat = maxat_list[cho]

cho2 = 0
emb_lev = [ 'kc', '1pl', '2pl', '3pl']
emb_choice = emb_lev[cho2]
 
while True:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader'])    
    # 解析输出并提取显卡内存剩余值
    memory_free_list = [int(memory.strip().split()[0]) for memory in output.decode().strip().split('\n')]   
    # 寻找内存剩余最大值
    max_memory_free = max(memory_free_list)
    # 获取最大内存剩余对应的显卡索引
    gpu_id = memory_free_list.index(max_memory_free)

# python scripts/train.py -m LSKT -d [assist09,assist17,algebra05,statics] -bs 32 -tbs 32 -p -cl --proj [-o output/LSKT_assist09] [--device cuda]
    # 输出当前剩余内存最大的显卡和剩余内存大小
    print(f"当前剩余内存最大的显卡是 GPU:{gpu_id+1}，剩余内存：{max_memory_free} MiB")
    if max_memory_free > 1800:
         if mode=='stage1':
               # 执行命令
               subprocess.run(['python', '/home/q22301200/current/KT_module/LSKT-main/scripts/train.py',
                                '-m', 'LSKT',
                                '-d',f'{choice}',  #assist09
                                '-bs', '16',
                                '-tbs','16',
                                '-p',
                                '-o',f'/home/q22301200/current/KT_module/LSKT-main/{choice}_result/{emb_choice}',
                                '--device',f'cuda:{gpu_id}',
                                # '--device',f'cuda:0',
                                # '--device','cpu',
                                '-emb','3pl'
                                ]) 

         # 停止脚本
         break  
    
    sleep_times = [300, 180, 480, 600]  # 对应 5 分钟、3 分钟、8 分钟和 10 分钟
    # 随机选择一个休眠时间
    # sleep_time = random.choice(sleep_times)
    sleep_time = 50
    # 进行休眠
    time.sleep(sleep_time) 



