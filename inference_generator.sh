#!/bin/bash

#SBATCH --job-name=inference_generator
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_generator.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-2

/home/panuthep/.conda/envs/iris/bin/python scripts/generate_cultural_examples.py