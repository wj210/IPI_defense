#!/bin/bash
echo "Running job with CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
cuda_path="cuda_visible_devices.txt"
if [ -f $cuda_path ]; then
  export CUDA_VISIBLE_DEVICES=$(cat $cuda_path)
  num_gpu=$(cat "$cuda_path" | tr ', ' '\n' | grep -c '[0-9]')
  echo "num gpus = $num_gpu"
else
  echo "cuda_visible_devices.txt file not found."
fi

for attack in SpclSpclSpcl_NaiveCompletion SpclSpclSpcl_None
do
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29551 train.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --data_path data/alpaca_data_cleaned.json \
  --bf16 True \
  --output_dir Qwen2.5-7B_$attack \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "Qwen2DecoderLayer" \
  --tf32 True \
  --attack $attack
done