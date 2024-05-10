#!/bin/bash

export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1

if [ $# -ne 8 ]; then
  echo "Usage: $0 <model_type> <dataset> <num_epochs> <batch_size> 
  <learning_rate> <num_gpus> <start_seed> <num_seeds>" 
  exit 1
fi

model_type=$1
dataset=$2
num_epochs=$3
batch_size=$4
lr=$5
num_gpus=$6
start_seed=$7
num_seeds=$8

quantizer_types=("lin" "lin+" "po2" "po2+")

for seed in $(seq $start_seed $((start_seed + num_seeds - 1)))
do
  echo "running seed $seed 🏃‍♂️..."
  # full precision training
  echo "running $model_type $dataset full precision 🏃‍♂️..."
  torchrun --standalone --nnodes=1 --nproc-per-node=$num_gpus train.py --model_type=$model_type --dataset=$dataset --quantizer_type=none --bits=4 --num_epochs=$num_epochs --batch_size=$batch_size --lr=$lr --seed=$seed

  # quantize aware training
  for quantizer_type in "${quantizer_types[@]}"
  do
    for bits in 3 4
    do
      echo "running $model_type $dataset QAT $quantizer_type $bits bits 🏃‍♂️..."
      torchrun --standalone --nnodes=1 --nproc-per-node=$num_gpus train.py --model_type=$model_type --dataset=$dataset --quantizer_type=$quantizer_type --bits=$bits --num_epochs=$num_epochs --batch_size=$batch_size --lr=$lr --seed=$seed
    done
  done
done



