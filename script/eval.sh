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

for model_name in Qwen2.5-7B_SpclSpclSpcl_None
do
  attack=('naive' 'ignore' 'escape_deletion' 'escape_separation' 'completion_other' 'completion_othercmb' 'completion_real' 'completion_realcmb' 'completion_close_2hash' 'completion_close_1hash' 'completion_close_0hash' 'completion_close_upper' 'completion_close_title' 'completion_close_nospace' 'completion_close_nocolon' 'completion_close_typo' 'completion_close_similar' 'hackaprompt')

  python eval.py --model_name_or_path $model_name --attack "${attack[@]}" --bz 64

done