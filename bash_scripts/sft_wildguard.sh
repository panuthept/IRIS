#!/bin/bash -l

#SBATCH --job-name=SFT_WildGuard
#SBATCH --output=sft_wildguard.out
#SBATCH --nodes=1
#SBATCH --partition=scads-a100
#SBATCH --account=scads
#SBATCH --gres=gpu:4
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

CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/sft_wildguard.py