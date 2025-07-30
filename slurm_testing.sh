#!/bin/bash

#SBATCH --job-name=slurm_testing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm_testing.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0

echo "Hello world, I am running on node $HOSTNAME"
sleep 10
date