cd /home/4T/wuhao_zjc/ab_new_document/AIChat_QingZhou/LLaMA-Factory

torchrun --nproc_per_node=1 src/train.py \
    --model_name_or_path "/home/4T/wuhao_zjc/ab_new_document/AIChat_QingZhou/LLaMA-Factory/models/qwen/Qwen15-7B-Chat" \
    --disable_sdpa\
    --dataset wechat_gqingzhou \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./output/qwen1.5-7b-lora-qingzhou \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500 \
    --ddp_timeout 1800