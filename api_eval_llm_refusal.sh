#!/bin/bash

judge_name=$1
load_path=$2
api_key=$3
api_base=$4

# Start VLLM server before running this script

echo "Evaluating Cultural Safety (Handwritten)..."
for lang in ta th tl ms in my vi
do
    echo "Cultural: $lang - Language: en"
    python scripts/eval_llm_refusal.py \
    --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl \
    --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/eval_refusal.jsonl \
    --model_name $judge_name \
    --api_key $api_key \
    --api_base $api_base 

    echo "Cultural: $lang - Language: $lang"
    python scripts/eval_llm_refusal.py \
    --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl \
    --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/eval_refusal.jsonl \
    --model_name $judge_name \
    --api_key $api_key \
    --api_base $api_base
done


echo "Evaluating Cultural Safety..."
for lang in en ta th tl ms in my vi
do
    echo "Cultural: $lang - Language: en"
    python scripts/eval_llm_refusal.py \
    --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl \
    --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/en/test/eval_refusal.jsonl \
    --model_name $judge_name \
    --api_key $api_key \
    --api_base $api_base

    echo "Cultural: $lang - Language: $lang"
    python scripts/eval_llm_refusal.py \
    --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl \
    --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/$lang/test/eval_refusal.jsonl \
    --model_name $judge_name \
    --api_key $api_key \
    --api_base $api_base
done


echo "Evaluating General Safety..."
for lang in en ta th tl ms in my vi
do
    echo "Language: $lang"
    python scripts/eval_llm_refusal.py \
    --load_path ./outputs/${load_path}/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl \
    --save_path ./outputs/${load_path}/SEASafeguardDataset/general/$lang/test/eval_refusal.jsonl \
    --model_name $judge_name \
    --api_key $api_key \
    --api_base $api_base
done