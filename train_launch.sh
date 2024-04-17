#!/bin/bash

export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1

if [ $# -ne 2 ]; then
  echo "Usage: $0 <model_type> <dataset>"
  exit 1
fi

model_type=$1
dataset=$2
quantizer_types=("lin" "lin+" "po2" "po2+")

# full precision training
echo "running $model_type $dataset full precision 🏃‍♂️..."
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py $model_type $dataset none 4

# quantize aware training
for quantizer_type in "${quantizer_types[@]}"
do
  for bits in 2 3 4
  do
    echo "running $model_type $dataset $quantizer_type $bits 🏃‍♂️..."
    torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py $model_type $dataset $quantizer_type $bits
  done
done