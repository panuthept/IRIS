#!/bin/bash

#SBATCH --job-name=inference_llamaguard_ta
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_llamaguard_ta.out
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0

/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-5000 \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language ta \
--max_tokens 3000 \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ta/test/all_prompts.jsonl