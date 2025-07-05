#!/bin/bash

load_path=$1

# # Start VLLM server before running this script
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 4
# CUDA_VISIBLE_DEVICES=4,5 vllm serve ToxicityPrompts/PolyGuard-Ministral --port 8001 --download_dir ./data/models --tensor_parallel_size 2
# CUDA_VISIBLE_DEVICES=6,7 vllm serve meta-llama/Llama-Guard-3-8B --port 8002 --download_dir ./data/models --tensor_parallel_size 2

# echo "Evaluating Cultural Safety (Handwritten)..."
# for lang in ta th tl ms in my vi
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/eval_llm_safe_response.py \
#     --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/all_prompts.jsonl \
#     --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/en/test/eval_safe_response.jsonl \
#     --model_name PolyGuard \
#     --model_name ToxicityPrompts/PolyGuard-Ministral \
#     --api_key EMPTY \
#     --api_base http://localhost:8001/v1 

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/eval_llm_safe_response.py \
#     --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/all_prompts.jsonl \
#     --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural_handwritten/$lang/test/eval_safe_response.jsonl \
#     --model_name LLMGuard \
#     --model_name google/gemma-3-27b-it \
#     --api_key EMPTY \
#     --api_base http://localhost:8000/v1
# done


echo "Evaluating Cultural Safety..."
echo "Cultural: en - Language: en"
python scripts/eval_llm_safe_response.py \
--load_path ./outputs/${load_path}/SEASafeguardDataset/en_cultural/en/test/all_prompts.jsonl \
--save_path ./outputs/${load_path}/SEASafeguardDataset/en_cultural/en/test/eval_safe_response.jsonl \
--model_name PolyGuard \
--model_name ToxicityPrompts/PolyGuard-Ministral \
--api_key EMPTY \
--api_base http://localhost:8001/v1

# for lang in ta th tl ms in my vi
# do
#     echo "Cultural: $lang - Language: en"
#     python scripts/eval_llm_safe_response.py \
#     --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/en/test/all_prompts.jsonl \
#     --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/en/test/eval_safe_response.jsonl \
#     --model_name PolyGuard \
#     --model_name ToxicityPrompts/PolyGuard-Ministral \
#     --api_key EMPTY \
#     --api_base http://localhost:8001/v1

#     echo "Cultural: $lang - Language: $lang"
#     python scripts/eval_llm_safe_response.py \
#     --load_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/$lang/test/all_prompts.jsonl \
#     --save_path ./outputs/${load_path}/SEASafeguardDataset/${lang}_cultural/$lang/test/eval_safe_response.jsonl \
#     --model_name LLMGuard \
#     --model_name google/gemma-3-27b-it \
#     --api_key EMPTY \
#     --api_base http://localhost:8000/v1
# done


# echo "Evaluating General Safety..."
# for lang in en ta th tl ms in my vi
# do
#     echo "Language: $lang"
#     python scripts/eval_llm_safe_response.py \
#     --load_path ./outputs/${load_path}/SEASafeguardDataset/general/$lang/test/all_prompts.jsonl \
#     --save_path ./outputs/${load_path}/SEASafeguardDataset/general/$lang/test/eval_safe_response.jsonl \
#     --safeguard_name LlamaGuard \
#     --model_name meta-llama/Llama-Guard-3-8B \
#     --api_key EMPTY \
#     --api_base http://localhost:8002/v1
# done