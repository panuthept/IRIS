#!/bin/bash

#SBATCH --job-name=inference_sealionguard
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_sealionguard.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:3
#SBATCH --nodelist=a3mega-a3meganodeset-0

echo "Evaluating Cultural Safety..."
for lang in ta th tl ms in my vi
do
    echo "Cultural: $lang - Language: en"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name SealionGuard \
    --model_name aisingapore/gemma2-9b-cpt-sea-lionv3-instruct \
    --checkpoint_path ./data/model_checkpoints/sealion_guard/checkpoint-5500 \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language en \
    --cultural $lang \
    --subset cultural_specific \
    --mixed_tasks_sample \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/SealionGuard-ckp5500/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name SealionGuard \
    --model_name aisingapore/gemma2-9b-cpt-sea-lionv3-instruct \
    --checkpoint_path ./data/model_checkpoints/sealion_guard/checkpoint-5500 \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --cultural $lang \
    --subset cultural_specific \
    --mixed_tasks_sample \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/SealionGuard-ckp5500/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
done


echo "Evaluating General Safety..."
for lang in en ta th tl ms in my vi
do
    echo "Language: $lang"
    /home/panuthep/.conda/envs/iris/bin/python scripts/inference_safeguard.py \
    --safeguard_name SealionGuard \
    --model_name aisingapore/gemma2-9b-cpt-sea-lionv3-instruct \
    --checkpoint_path ./data/model_checkpoints/sealion_guard/checkpoint-5500 \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --subset general \
    --mixed_tasks_sample \
    --max_tokens 3000 \
    --disable_logitlens \
    --output_path ./outputs/SealionGuard-ckp5500/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done