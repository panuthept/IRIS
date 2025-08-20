#!/bin/bash

#SBATCH --job-name=inference_sealionguard_gemma_full
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inference_sealionguard_gemma_full.out
#SBATCH --time=1440:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodelist=a3mega-a3meganodeset-0

safeguard_name=$1
model_name=$2
output_name=$3
api_key=$4
api_base=$5


# echo "Starting VLLM server..."
# vllm serve $model_name --download_dir ./data/models --tensor_parallel_size $gpu_nums


echo "Evaluating Cultural Safety..."
echo "Cultural: en - Language: en"
python scripts/inference_safeguard.py \
--safeguard_name $safeguard_name \
--model_name $model_name \
--api_key $api_key \
--api_base $api_base \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en \
--cultural en \
--subset cultural_specific \
--mixed_tasks_sample \
--max_tokens 8000 \
--disable_logitlens \
--sensitive_as_harmful \
--output_path ./outputs/${output_name}/SEASafeguardDataset/en_cultural/en/test/all_prompts.jsonl


echo "Evaluating General Safety..."
for lang in en
do
    echo "Language: $lang"
    python scripts/inference_safeguard.py \
    --safeguard_name $safeguard_name \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --dataset_name SEASafeguardDataset \
    --dataset_split test \
    --language $lang \
    --subset general \
    --mixed_tasks_sample \
    --max_tokens 8000 \
    --disable_logitlens \
    --sensitive_as_harmful \
    --output_path ./outputs/${output_name}/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done