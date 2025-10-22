# QuickStart for LLaMA

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

---

### ✅ 前提确认
- 你已清洗出 `other_to_you.jsonl`，格式为：
  ```json
  {"input": "对方消息", "output": "你的回复"}
  ```
- 服务器：2×V100（32GB/卡，共64GB），Linux，支持 `bf16`
- 网络：国内，优先使用 **ModelScope**（而非 Hugging Face）

---

## 🧩 阶段 1：选择基础模型（Backbone）

✅ **推荐**：`Qwen/Qwen1.5-7B-Chat`（中文对话最强，ModelScope 官方支持）

> ModelScope 模型 ID：`qwen/Qwen1.5-7B-Chat`

---

## 🛠 阶段 2：准备环境 & 数据格式

### 1. 安装 LLaMA-Factory（启用 ModelScope 支持）
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 2. 启用 ModelScope Hub（国内加速）
```bash
export USE_MODELSCOPE_HUB=1
```

> 此后所有 `model_name_or_path` 可直接使用 ModelScope ID，如 `qwen/Qwen1.5-7B-Chat`

### 3. 注册你的私有数据集

#### (1) 将数据放入目录（例如）：
```bash
mkdir -p data/my_chat/
cp /your/path/other_to_you.jsonl data/my_chat/
```

#### (2) 编辑 `data/dataset_info.json`，**新增条目**：
```json
{
  "my_chat": {
    "file_name": "my_chat/other_to_you.jsonl",
    "format": "alpaca"
  }
}
```

> ✅ 格式说明：`alpaca` 对应 `{"input": "...", "output": "..."}`，完全匹配你的数据。

---

## 🔥 阶段 3：LoRA 微调（使用 LLaMA-Factory 官方 CLI）

### 1. 创建训练配置：`examples/train_lora/qwen1_5_7b_lora_sft.yaml`

```yaml
model_name_or_path: qwen/Qwen1.5-7B-Chat
template: qwen
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: all  # 或指定 q_proj,v_proj 等

dataset: my_chat
dataset_dir: data
split: train
max_samples: -1
overwrite_cache: true

per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 3e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03

bf16: true
ddp_timeout: 18000
logging_steps: 10
save_steps: 500
output_dir: saves/qwen1_5_7b/lora/sft
```

### 2. 启动训练（自动多卡，无需手动 torchrun）

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen1_5_7b_lora_sft.yaml
```

> ✅ LLaMA-Factory 内部自动调用 `torchrun`，你只需指定 GPU 即可。  
> 💡 显存占用：~28GB/卡（V100 32GB 足够）

---

## 🚀 阶段 4：合并 LoRA + 本地 API 部署

### 1. 合并 LoRA 到完整模型

> ⚠️ 注意：合并时**不能使用量化模型**，必须用原始 BF16 模型。

创建 `examples/merge_lora/qwen1_5_7b_lora_sft.yaml`：
```yaml
model_name_or_path: qwen/Qwen1.5-7B-Chat
adapter_name_or_path: saves/qwen1_5_7b/lora/sft
template: qwen
export_dir: models/qwen1_5_7b_finetuned
export_size: 2  # 分片保存（可选）
export_device: cpu  # 节省 GPU 显存
```

执行合并：
```bash
llamafactory-cli export examples/merge_lora/qwen1_5_7b_lora_sft.yaml
```

输出目录：`models/qwen1_5_7b_finetuned/`（含完整 `pytorch_model.bin`）

---

### 2. 启动 OpenAI 兼容 API（使用 vLLM，高性能）

```bash
# 安装 vLLM（如未安装）
pip install vllm

# 启动服务（自动使用 2 张 V100）
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen1_5_7b_finetuned \
    --tensor-parallel-size 2 \
    --port 8000 \
    --dtype bfloat16
```

> ✅ 支持并发、流式输出、OpenAI 标准接口

### 3. 测试调用（Python）

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="qwen1.5-7b-finetuned",
    messages=[{"role": "user", "content": "你还记得我们第一次见面吗？"}],
    temperature=0.7,
    max_tokens=256
)
print(resp.choices[0].message.content)
```

---

## 🌐 阶段 5（可选扩展）：Web 界面 or RAG

### 🔹 快速 Web 聊天界面（Gradio）
```python
# web_demo.py
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def predict(message, history):
    messages = [{"role": "user", "content": msg} for msg, _ in history]
    messages.append({"role": "user", "content": message})
    resp = client.chat.completions.create(model="qwen1.5-7b-finetuned", messages=messages)
    return resp.choices[0].message.content

gr.ChatInterface(predict).launch(server_name="0.0.0.0", server_port=7860)
```

运行：
```bash
python web_demo.py
```
访问：`http://<your-server>:7860`

---

### 🔹 RAG 增强（检索历史聊天）
- 将未用于训练的聊天记录（如 `you_to_other.jsonl`）导入 **ChromaDB**
- 在 API 前加检索模块，拼接上下文
- 可用 LLaMA-Factory 的 `rag` 模板或自定义 pipeline

---

## ✅ 最终流程图（更新版）

```
原始微信JSON
     ↓
[数据清洗] → other_to_you.jsonl（✅ 已完成）
     ↓
[注册数据集] → data/dataset_info.json + data/my_chat/
     ↓
[启用ModelScope] → export USE_MODELSCOPE_HUB=1
     ↓
[LoRA微调] → llamafactory-cli train ...（2×V100）
     ↓
[合并模型] → llamafactory-cli export ...
     ↓
[部署API] → vLLM OpenAI server（端口 8000）
     ↓
[使用] → Python / curl / Web / RAG
```

---

## 📌 关键提醒

1. **不要用 `torchrun` 直接调脚本**：LLaMA-Factory 推荐使用 `llamafactory-cli`，它已封装 DDP、DeepSpeed、Ray 等后端。
2. **ModelScope 优先**：国内下载快，避免 HF 网络问题。
3. **LoRA 合并必须用原始模型**：不能是量化版（如 GPTQ/AWQ）。
4. **V100 不支持 FlashAttention-2**：训练时请关闭 `flash_attn`（默认已适配）。

---

需要我为你生成：
- 完整的 `qwen1_5_7b_lora_sft.yaml`？
- 一键训练+合并+部署脚本？
- Gradio 前端 + RAG 示例？

随时告诉我！你现在可以安全进入 **阶段 2（环境准备）** 了。