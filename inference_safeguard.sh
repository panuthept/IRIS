#!/bin/bash

#SBATCH --job-name=inference_llamaguard_culturals
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_llamaguard_culturals.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0

echo "Evaluating Cultural Safety..."
# for lang in ta th tl ms in my vi
for lang in my
do
    echo "Cultural: $lang - Language: en"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name LlamaGuard \
    --model_name meta-llama/Llama-Guard-3-8B \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language en \
    --cultural $lang \
    --subset cultural_specific \
    --mixed_tasks_sample \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/Finetuned_LlamaGuard8B/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name LlamaGuard \
    --model_name meta-llama/Llama-Guard-3-8B \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --cultural $lang \
    --subset cultural_specific \
    --mixed_tasks_sample \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/Finetuned_LlamaGuard8B/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
done


echo "Evaluating General Safety..."
for lang in en ta th tl ms in my vi
do
    echo "Language: $lang"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name LlamaGuard \
    --model_name meta-llama/Llama-Guard-3-8B \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --subset general \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/Finetuned_LlamaGuard8B/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done