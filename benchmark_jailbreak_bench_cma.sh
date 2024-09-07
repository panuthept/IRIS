#!/bin/bash -l

#SBATCH --job-name=IRIS_JailbreakBench_CAM
#SBATCH --output=benchmark_jailbreak_bench_cma.out
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
CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/benchmark_jailbreak_bench.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --intervention \
    --intervention_layers 19 20 21 22 \
    --max_tokens 512 