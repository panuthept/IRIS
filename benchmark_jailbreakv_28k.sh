#!/bin/bash -l

#SBATCH --job-name=IRIS
#SBATCH --output=benchmark.out
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

echo "Evaluating Qwen/Qwen2-0.5B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --max_tokens 512 \

echo "Evaluating Qwen/Qwen2-1.5B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name Qwen/Qwen2-1.5B-Instruct \
    --max_tokens 512 \

echo "Evaluating Qwen/Qwen2-7B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --max_tokens 512 \

echo "Evaluating google/gemma-2-2b-it:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name google/gemma-2-2b-it \
    --max_tokens 512 \

echo "Evaluating google/gemma-2-9b-it:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name google/gemma-2-9b-it \
    --max_tokens 512 \

echo "Evaluating meta-llama/Meta-Llama-3.1-8B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreakv_28k.py.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max_tokens 512 \