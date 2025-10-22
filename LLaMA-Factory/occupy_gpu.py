import torch
import time
import argparse

def occupy_gpu(gpu_ids, memory_mb):
    tensors = []
    for gpu_id in gpu_ids:
        torch.cuda.set_device(gpu_id)
        # 计算所需元素数量：每个 float32 元素占 4 字节
        num_elements = (memory_mb * 1024 * 1024) // 4
        print(f"Allocating {memory_mb} MB on cuda:{gpu_id}...")
        tensor = torch.empty(num_elements, dtype=torch.float32, device=f'cuda:{gpu_id}')
        tensors.append(tensor)  # 保留引用，防止被释放
    return tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[4, 5], help="GPU IDs to occupy (e.g., --gpus 4 5)")
    parser.add_argument("--memory_mb", type=int, default=30000, help="Memory to allocate per GPU in MB")
    parser.add_argument("--duration", type=int, default=3600, help="Hold memory for N seconds (default: 3600s = 1 hour)")
    args = parser.parse_args()

    print(f"Occupying GPUs {args.gpus} with {args.memory_mb} MB each...")
    tensors = occupy_gpu(args.gpus, args.memory_mb)

    print("Memory allocated. Holding...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted. Releasing memory...")
    finally:
        del tensors
        torch.cuda.empty_cache()
        print("Memory released.")