#!/bin/bash

model_name=$1
output_name=$2
api_key=$3
api_base=$4

# Start VLLM server before running this script

# echo "Evaluating Cultural Safety (Handwritten)..."
# for lang in my
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/inference_llm.py \
#     --model_name $model_name \
#     --api_key $api_key \
#     --api_base $api_base \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language en \
#     --cultural $lang \
#     --subset cultural_specific_handwritten \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/inference_llm.py \
#     --model_name $model_name \
#     --api_key $api_key \
#     --api_base $api_base \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language $lang \
#     --cultural $lang \
#     --subset cultural_specific_handwritten \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl
# done


# echo "Evaluating Cultural Safety..."
# for lang in my
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/inference_llm.py \
#     --model_name $model_name \
#     --api_key $api_key \
#     --api_base $api_base \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language en \
#     --cultural $lang \
#     --subset cultural_specific \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/inference_llm.py \
#     --model_name $model_name \
#     --api_key $api_key \
#     --api_base $api_base \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language $lang \
#     --cultural $lang \
#     --subset cultural_specific \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
# done


# echo "Evaluating General Safety..."
# for lang in my
# do
#     echo "Language: $lang"
#     python scripts/inference_llm.py \
#     --model_name $model_name \
#     --api_key $api_key \
#     --api_base $api_base \
#     --dataset_name SEASafeguardDataset \
#     --dataset_split test \
#     --language $lang \
#     --subset general \
#     --output_path ./outputs/${output_name}/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
# done


# Safeguard-advance-inform (LlamaGuard)
echo "Evaluating Cultural Safety (Handwritten)..."
for lang in my
do
    echo "Cultural: $lang - Language: en"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGuard/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LlamaGuard/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGuard/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LlamaGuard/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl
done

echo "Evaluating Cultural Safety..."
for lang in my
do
    echo "Cultural: $lang - Language: en"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGuard/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LlamaGuard/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGuard/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LlamaGuard/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
done

echo "Evaluating General Safety..."
for lang in my
do
    echo "Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGuard/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LlamaGuard/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done


# Safeguard-advance-inform (LLMGuard-Llama3.3-70B)
echo "Evaluating Cultural Safety (Handwritten)..."
for lang in my
do
    echo "Cultural: $lang - Language: en"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl
done

echo "Evaluating Cultural Safety..."
for lang in my
do
    echo "Cultural: $lang - Language: en"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl

    echo "Cultural: $lang - Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LlamaGLLMGuard-Llama3.3-70Buard/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LLMGuard-Llama3.3-70B/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl
done

echo "Evaluating General Safety..."
for lang in my
do
    echo "Language: $lang"
    python scripts/inference_llm_with_safeguard.py \
    --safeguard_model advance_inform \
    --model_name $model_name \
    --api_key $api_key \
    --api_base $api_base \
    --input_path ./outputs/LLMGuard-Llama3.3-70B/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl \
    --output_path ./outputs/${output_name}-LLMGuard-Llama3.3-70B/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl
done