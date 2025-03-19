#!/bin/bash

#SBATCH --job-name=inference_shieldgemma_th_culture
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_shieldgemma_th_culture.out
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-2

echo "Language: en"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name ShieldGemma \
--model_name google/shieldgemma-9b \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en \
--cultural th \
--subset cultural_specific \
--max_tokens 3000 \
--disable_logitlens \
--output_path ./outputs/ShieldGemma9B/SEASafeguardDataset/en/test/all_prompts.jsonl

echo "Language: th"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name ShieldGemma \
--model_name google/shieldgemma-9b \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language th \
--cultural th \
--subset cultural_specific \
--max_tokens 3000 \
--disable_logitlens \
--output_path ./outputs/ShieldGemma9B/SEASafeguardDataset/th/test/all_prompts.jsonl

# echo "Language: ta"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural ta \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ta/test/all_prompts.jsonl

# echo "Language: th"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural th \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th/test/all_prompts.jsonl

# echo "Language: tl"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural tl \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/tl/test/all_prompts.jsonl

# echo "Language: ms"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural ms \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ms/test/all_prompts.jsonl

# echo "Language: in"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural in \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/in/test/all_prompts.jsonl

# echo "Language: my"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural my \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/my/test/all_prompts.jsonl

# echo "Language: vi"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --cultural vi \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/vi/test/all_prompts.jsonl