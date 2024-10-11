#!/bin/bash -l

#SBATCH --job-name=SFT_WildGuard_V100
#SBATCH --output=sft_wildguard_v100.out
#SBATCH --nodes=1
#SBATCH --partition=scads
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

CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/sft_wildguard.py \
--model_name facebook/opt-350m \
--max_seq_length 2048 \
--batch_size 1 \
--gradient_accumulation_steps 128 \
--output_dir ./finetuned_models/sft_wildguard_opt_350m