#!/bin/bash -l

#SBATCH --job-name=IRIS
#SBATCH --output=benchmark_multigpu.out
#SBATCH --nodes=1
#SBATCH --partition=scads-a100
#SBATCH --account=scads
#SBATCH --gres=gpu:2
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


echo "Evaluating Qwen/Qwen2-72B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_instruction_induction.py \
    --model_name Qwen/Qwen2-72B-Instruct \
    --max_tokens 512 \

echo "Evaluating meta-llama/Meta-Llama-3.1-70B-Instruct:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_instruction_induction.py \
    --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --max_tokens 512 \