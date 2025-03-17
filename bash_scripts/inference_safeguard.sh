#!/bin/bash

#SBATCH --job-name=inference_llamaguard
#SBATCH --output=inference_llamaguard.out
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-4

/home/panuthep/.conda/envs/iris/bin/python scripts/inference_wildguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-5000 \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/en/test/all_prompts.jsonl