#!/bin/bash -l

#SBATCH --job-name=IRIS
#SBATCH --output=inference.out
#SBATCH --nodes=1
#SBATCH --partition=scads-a100
#SBATCH --account=scads
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --time=5-00:00:0
#SBATCH --cpus-per-task=10

port=$(shuf -i 6000-9999 -n 1)
USER=$(whoami)
node=$(hostname -s)

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

model_name=$1
max_tokens=$2

echo "Evaluating $model_name with $max_tokens max tokens on JailbreakBench dataset:"

echo "Original benign set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --dataset_split benign

echo "Original harmful set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --dataset_split harmful

echo "GCG harmful set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --attack_engine GCG \
    --dataset_split harmful

echo "JBC harmful set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --attack_engine JBC \
    --dataset_split harmful

echo "PAIR harmful set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --attack_engine PAIR \
    --dataset_split harmful

echo "Random search harmful set:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_jailbreak_bench.py \
    --model_name $model_name \
    --max_tokens $max_tokens \
    --attack_engine prompt_with_random_search \
    --dataset_split harmful