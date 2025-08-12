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


# model_name=Qwen3-8B_ISE
# attack=('naive' 'ignore' 'escape_deletion' 'escape_separation' 'completion_other' 'completion_othercmb' 'completion_real' 'completion_realcmb' 'completion_close_2hash' 'completion_close_1hash' 'completion_close_0hash' 'completion_close_upper' 'completion_close_title' 'completion_close_nospace' 'completion_close_nocolon' 'completion_close_typo' 'completion_close_similar' 'hackaprompt')
# # attack=('hackaprompt')

# python test_on_struq.py --model $model_name --attack "${attack[@]}" --batch_size 64 --embedding_type 'ise' --base_model 'Qwen3-8B'

## alpaca eval
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=29712 get_alpaca_outputs.py --data-path data/tatsu-lab/alpaca_farm/eval.json --use-input True --model Qwen3-8B-ASIDE --embedding-type forward_rot --batch-size 64

torchrun --nproc_per_node=1 --master_port=29712 get_alpaca_outputs.py --data-path data/tatsu-lab/alpaca_farm/eval.json --use-input True --model Qwen3-8B-ISE --embedding-type ise --batch-size 64

# IS_ALPACA_EVAL_2=False alpaca_eval --model_outputs ../../data/tatsu-lab/alpaca_farm/Qwen2.5-7B_forward_rot_train_checkpoints_SFTv70_from_inst_run_ASIDE_last__l-1_s42.json
