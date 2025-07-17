#!/bin/bash

#SBATCH --job-name=inference_sealionguard_gemma_full
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_sealionguard_gemma_full.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0

safeguard_name=$1
output_name=$2
api_key=$3
endpoint=$4


# echo "Evaluating Cultural Safety (Handwritten)..."
# for lang in ta th tl ms in my vi
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/inference_safeguard.py \
#     --safeguard_name $safeguard_name \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language en \
#     --cultural $lang \
#     --subset cultural_specific_handwritten \
#     --mixed_tasks_sample \
#     --max_tokens 8000 \
#     --disable_logitlens \
#     --sensitive_as_harmful \
#     --api_key $api_key \
#     --endpoint $endpoint \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/inference_safeguard.py \
#     --safeguard_name $safeguard_name \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language $lang \
#     --cultural $lang \
#     --subset cultural_specific_handwritten \
#     --mixed_tasks_sample \
#     --max_tokens 8000 \
#     --disable_logitlens \
#     --sensitive_as_harmful \
#     --api_key $api_key \
#     --endpoint $endpoint \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl
# done


# echo "Evaluating Cultural Safety..."
# for lang in en ta th tl ms in my vi
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/inference_safeguard.py \
#     --safeguard_name $safeguard_name \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language en \
#     --cultural $lang \
#     --subset cultural_specific \
#     --mixed_tasks_sample \
#     --max_tokens 8000 \
#     --disable_logitlens \
#     --sensitive_as_harmful \
#     --api_key $api_key \
#     --endpoint $endpoint \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/inference_safeguard.py \
#     --safeguard_name $safeguard_name \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language $lang \
#     --cultural $lang \
#     --subset cultural_specific \
#     --mixed_tasks_sample \
#     --max_tokens 8000 \
#     --disable_logitlens \
#     --sensitive_as_harmful \
#     --api_key $api_key \
#     --endpoint $endpoint \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
# done


echo "Evaluating General Safety..."
for lang in ms in my vi
# for lang in en ta th tl ms in my vi
do
    echo "Language: $lang"
    python scripts/inference_safeguard.py \
    --safeguard_name $safeguard_name \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --subset general \
    --mixed_tasks_sample \
    --max_tokens 8000 \
    --disable_logitlens \
    --sensitive_as_harmful \
    --api_key $api_key \
    --endpoint $endpoint \
    --output_path ./outputs/${output_name}/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done