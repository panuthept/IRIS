#!/bin/bash -l

#SBATCH --job-name=IRIS
#SBATCH --output=inference_multigpu.out
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

model_name=$1
max_tokens=$2

echo "Evaluating $model_name with $max_tokens max tokens on InstructionIndutionDataset dataset:"

CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/iris/bin/python scripts/inference_instruction_induction.py \
    --task_name sentiment \
    --model_name $model_name \
    --max_tokens $max_tokens \