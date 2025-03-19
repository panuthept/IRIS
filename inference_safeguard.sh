#!/bin/bash

#SBATCH --job-name=inference_finetuned_llamaguard_en_ta
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_finetuned_llamaguard_en_ta.out
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-2

echo "Language: en"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
--dataset_name SEASafeguardDataset \
--dataset_split train \
--language en \
--subset general \
--max_tokens 3000 \
--max_samples 1000 \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/en/train/all_prompts.jsonl

echo "Language: ta"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
--dataset_name SEASafeguardDataset \
--dataset_split train \
--cultural ta \
--subset general \
--max_tokens 3000 \
--max_samples 1000 \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ta/train/all_prompts.jsonl

# echo "Language: th"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural th \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th/train/all_prompts.jsonl

# echo "Language: tl"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural tl \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/tl/train/all_prompts.jsonl

# echo "Language: ms"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural ms \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ms/train/all_prompts.jsonl

# echo "Language: in"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural in \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/in/train/all_prompts.jsonl

# echo "Language: my"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural my \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/my/train/all_prompts.jsonl

# echo "Language: vi"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split train \
# --cultural vi \
# --subset general \
# --max_tokens 3000 \
# --max_samples 1000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/vi/train/all_prompts.jsonl