#!/bin/bash

#SBATCH --job-name=inference_finetuned_llamaguard_culturals
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_finetuned_llamaguard_culturals.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0


for lang in ta th tl ms in my vi
do
    echo "Cultural: $lang - Language: en"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name LlamaGuard \
    --model_name meta-llama/Llama-Guard-3-8B \
    --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language en \
    --cultural $lang \
    --subset cultural_specific \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name LlamaGuard \
    --model_name meta-llama/Llama-Guard-3-8B \
    --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --cultural $lang \
    --subset cultural_specific \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
done

# echo "Cultural: th - Language: en"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language en \
# --cultural th \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th_cultural/en/test/all_prompts.jsonl

# echo "Cultural: th - Language: th"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language th \
# --cultural th \
# --subset cultural_specific \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th_cultural/th/test/all_prompts.jsonl




# echo "Language: en"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language en \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/en/test/all_prompts.jsonl

# echo "Language: ta"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language ta \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ta/test/all_prompts.jsonl

# echo "Language: th"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language th \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/th/test/all_prompts.jsonl

# echo "Language: tl"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language tl \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/tl/test/all_prompts.jsonl

# echo "Language: ms"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language ms \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/ms/test/all_prompts.jsonl

# echo "Language: in"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language in \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/in/test/all_prompts.jsonl

# echo "Language: my"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language my \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/my/test/all_prompts.jsonl

# echo "Language: vi"
# /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
# --safeguard_name LlamaGuard \
# --model_name meta-llama/Llama-Guard-3-8B \
# --checkpoint_path ./data/model_checkpoints/finetuned_llamaguard/checkpoint-11500 \
# --dataset_name SEASafeguardDataset \
# --dataset_split test \
# --language vi \
# --subset general \
# --max_tokens 3000 \
# --disable_logitlens \
# --output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/vi/test/all_prompts.jsonl