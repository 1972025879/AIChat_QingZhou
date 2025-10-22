# author: muzhan
import os
import sys
import time
import pynvml

def get_all_gpu_memory():
    """返回所有 GPU 的显存使用量（MiB）列表"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_list = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used // (1024 * 1024)  # 转为 MiB
        memory_list.append(used_mb)
    pynvml.nvmlShutdown()
    return memory_list

def wait_for_free_gpus(required_gpu_count=2, memory_threshold=10000, interval=2):
    """
    等待直到有至少 required_gpu_count 个 GPU 的显存使用量 <= memory_threshold。
    返回这些空闲 GPU 的索引列表。
    """
    print(f"Waiting for {required_gpu_count} free GPU(s) (<= {memory_threshold} MiB used)...")
    count = 0
    while True:
        try:
            memories = get_all_gpu_memory()
        except Exception as e:
            print(f"\nError reading GPU info: {e}")
            time.sleep(interval)
            continue

        # 找出所有空闲的 GPU（显存使用 <= 阈值）
        free_gpus = [i for i, mem in enumerate(memories) if mem <= memory_threshold]

        if len(free_gpus) >= required_gpu_count:
            selected_gpus = free_gpus[:required_gpu_count]  # 取前 required_gpu_count 个
            print(f"\nFound enough free GPUs: {selected_gpus}")
            return selected_gpus

        print(f"count: {count}", end='\r')
        count += 1

        # 显示状态
        status_str = " | ".join([f"GPU{i}:{mem}MiB" for i, mem in enumerate(memories)])
        sys.stdout.write(f'\rWaiting... {status_str} | Free: {free_gpus}')
        sys.stdout.flush()
        time.sleep(interval)

import subprocess
import sys

def run_cmd_on_gpus(cmd, gpu_ids):
    gpu_str = ",".join(map(str, gpu_ids))
    env = os.environ.copy()  # 复制当前环境
    env["CUDA_VISIBLE_DEVICES"] = gpu_str  # 设置变量
    print(f"[INFO] Setting CUDA_VISIBLE_DEVICES={gpu_str}")
    print(f"[CMD] {cmd}")
    
    # 显式传入 env，确保子进程使用它
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main(cmd='bash ./train_2.sh', required_gpu_count=2, memory_threshold=10000, interval=2):
    free_gpus = wait_for_free_gpus(
        required_gpu_count=required_gpu_count,
        memory_threshold=memory_threshold,
        interval=interval
    )
    run_cmd_on_gpus(cmd, free_gpus)

if __name__ == '__main__':
    # 你可以在这里自定义参数
    main(
        cmd='bash ./train.sh',
        required_gpu_count=1,
        memory_threshold=6000,  # MiB
        interval=2
    )