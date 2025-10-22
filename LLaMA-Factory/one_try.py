from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(
    "/home/4T/wuhao_zjc/ab_new_document/AIChat_QingZhou/LLaMA-Factory/models/qwen/Qwen15-7B-Chat",
    trust_remote_code=True
)

ds = load_dataset(
    "json",
    data_files="/home/4T/wuhao_zjc/ab_new_document/AIChat_QingZhou/Dataset/json/wxid_wg6cp8uh982s22/you_to_other.jsonl",
    split="train"
)

def is_valid(example):
    try:
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(text, truncation=False)["input_ids"]
        return len(tokens) > 0
    except Exception as e:
        print(f"Error on sample: {example} | {e}")
        return False

valid_count = sum(1 for ex in ds if is_valid(ex))
print(f"Valid after Qwen template: {valid_count} / {len(ds)}")