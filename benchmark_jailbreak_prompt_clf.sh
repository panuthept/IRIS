#!/bin/bash -l

#SBATCH --job-name=IRIS_Jailbreak_Prompt_CLF
#SBATCH --output=benchmark_jailbreak_prompt_clf.out
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

echo "Evaluating meta-llama/LlamaGuard-7b on JailbreakBench:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreak_prompt_clf.py \
    --model_name meta-llama/LlamaGuard-7b \
    --benchmark_name jailbreak_bench

echo "Evaluating meta-llama/LlamaGuard-7b on JailbreaKV28k:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreak_prompt_clf.py \
    --model_name meta-llama/LlamaGuard-7b \
    --benchmark_name jailbreakv_28k

echo "Evaluating meta-llama/LlamaGuard-7b on WildGuardMix:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreak_prompt_clf.py \
    --model_name meta-llama/LlamaGuard-7b \
    --benchmark_name wildguardmix

echo "Evaluating meta-llama/LlamaGuard-7b on XSTest:"
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreak_prompt_clf.py \
    --model_name meta-llama/LlamaGuard-7b \
    --benchmark_name xstest