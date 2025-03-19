#!/bin/bash

#SBATCH --job-name=inference_finetuned_llamaguard_en_ta
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference__finetuned_llamaguard_en_ta.out
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-2

echo "Language: en"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en \
--max_tokens 3000 \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/en/test/all_prompts.jsonl

echo "Language: ta"
/home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language ta \
--max_tokens 3000 \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ta/test/all_prompts.jsonl

# echo "Language: th"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language th \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th/test/all_prompts.jsonl

# echo "Language: tl"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language tl \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/tl/test/all_prompts.jsonl

# echo "Language: ms"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language ms \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ms/test/all_prompts.jsonl

# echo "Language: in"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language in \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/in/test/all_prompts.jsonl

# echo "Language: my"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language my \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/my/test/all_prompts.jsonl

# echo "Language: vi"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-3500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language vi \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/vi/test/all_prompts.jsonl