# vLLM version: 0.8.5.post1
# huggingface-cli login --token $1 --add-to-git-credential

# vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 8
# vllm serve meta-llama/Llama-Guard-4-12B --port 8000 --download_dir ./data/models --tensor_parallel_size 8

export TRANSFORMERS_CACHE="/shared/panuthep/cache/huggingface/transformers"
export HF_DATASETS_CACHE="/shared/panuthep/cache/huggingface/datasets"
export HF_HOME="/shared/panuthep/cache/huggingface"
export VLLM_CACHE_ROOT="/shared/panuthep/cache/vllm"

# 1024 (DONE)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/gemma-3-4b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh LLMGuard google/gemma-3-4b-it LLMGuard-Gemma3-4B EMPTY http://localhost:8000/v1

# 1024 (DONE)
vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 8
sh api_inference_safeguard.sh LLMGuard google/gemma-3-27b-it LLMGuard-Gemma3-27B EMPTY http://localhost:8000/v1

# 1024 (DONE)
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh Llama31 meta-llama/Llama-3.1-8B-Instruct LLMGuard-Llama3.1-8B EMPTY http://localhost:8001/v1

# 1024 (DONE)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh Llama31 meta-llama/Llama-3.2-3B-Instruct LLMGuard-Llama3.2-3B EMPTY http://localhost:8000/v1

# 1024 (DONE)
vllm serve Meta-Llama-3.1-70B-Instruct --port 8000 --tensor_parallel_size 8
sh api_inference_safeguard.sh Llama31 Meta-Llama-3.1-70B-Instruct LLMGuard-Llama3.1-70B EMPTY http://localhost:8000/v1

vllm serve Llama-3.3-70B-Instruct --port 8000 --tensor_parallel_size 8
sh api_inference_safeguard.sh Llama31 Llama-3.3-70B-Instruct LLMGuard-Llama3.3-70B EMPTY http://localhost:8000/v1

# 1024 (DONE)
echo "Evaluating ShieldGemma-2B..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/shieldgemma-2b --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh ShieldGemma google/shieldgemma-2b ShieldGemma-2B EMPTY http://localhost:8000/v1 > inference_safeguard_shieldgemma_2b_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating PolyGuard-Qwen-Smol..."
CUDA_VISIBLE_DEVICES=4,5,6,7 sh inference_safeguard.sh PolyGuard ToxicityPrompts/PolyGuard-Qwen-Smol PolyGuard-Qwen-Smol

# 1024 (DONE)
echo "Evaluating LlamaGuard-3-1B..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-Guard-3-1B --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh LlamaGuard meta-llama/Llama-Guard-3-1B LlamaGuard-1B EMPTY http://localhost:8001/v1 > inference_safeguard_llamaguard_1b_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating ShieldGemma..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve google/shieldgemma-9b --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh ShieldGemma google/shieldgemma-9b ShieldGemma EMPTY http://localhost:8000/v1 > inference_safeguard_shieldgemma_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating PolyGuard-Qwen..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ToxicityPrompts/PolyGuard-Qwen --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh PolyGuard ToxicityPrompts/PolyGuard-Qwen PolyGuard-Qwen EMPTY http://localhost:8001/v1 > inference_safeguard_polyguard_qwen_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating PolyGuard-Ministral..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ToxicityPrompts/PolyGuard-Ministral --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh PolyGuard ToxicityPrompts/PolyGuard-Ministral PolyGuard-Ministral EMPTY http://localhost:8000/v1 > inference_safeguard_polyguard_ministral_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating LlamaGuard-3..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Llama-Guard-3-8B --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh LlamaGuard meta-llama/Llama-Guard-3-8B LlamaGuard EMPTY http://localhost:8001/v1 > inference_safeguard_llamaguard_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating LlamaGuard-4..."
CUDA_VISIBLE_DEVICES=0,1,2,3 sh inference_safeguard.sh LlamaGuard4 meta-llama/Llama-Guard-4-12B LlamaGuard4

# 1024 (DONE)
echo "Evaluating Llama-SealionGuard..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-SEA-LION-Guard --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard Llama-SealionGuard EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Llama-SealionGuard-EN-Cultural..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve aisingapore/Llama-SEA-LION-Guard-EN-Cultural --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-EN-Cultural Llama-SealionGuard-EN-Cultural EMPTY http://localhost:8001/v1 > inference_safeguard_llama_sealionguard_en_cultural_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Llama-SEA-LION-Guard-THEN-Cultural..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-SEA-LION-Guard-THEN-Cultural --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-THEN-Cultural Llama-SealionGuard-THEN-Cultural EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_then_cultural_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Llama-SEA-LION-Guard-TH-Cultural..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve aisingapore/Llama-SEA-LION-Guard-TH-Cultural --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-TH-Cultural Llama-SealionGuard-TH-Cultural EMPTY http://localhost:8001/v1 > inference_safeguard_llama_sealionguard_th_cultural_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Llama-SEA-LION-Guard-Cheat-General-Cultural..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-SEA-LION-Guard-Cheat-General-Cultural --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-Cheat-General-Cultural Llama-SEA-LION-Guard-Cheat-General-Cultural EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_cheat_general_cultural_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Gemma-SealionGuard..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve aisingapore/Gemma-SEA-LION-Guard --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh GemmaSealionGuard aisingapore/Gemma-SEA-LION-Guard Gemma-SealionGuard EMPTY http://localhost:8001/v1 > inference_safeguard_gemma_sealionguard_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Gemma-SealionGuard-EN-Cultural..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Gemma-SEA-LION-Guard-EN-Cultural --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh GemmaSealionGuard aisingapore/Gemma-SEA-LION-Guard-EN-Cultural Gemma-SealionGuard-EN-Cultural EMPTY http://localhost:8000/v1 > inference_safeguard_gemma_sealionguard_en_cultural_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating Llama-SEA-LION-Guard-Cultural-ENv2..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve aisingapore/Llama-SEA-LION-Guard-Cultural-ENv2 --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-Cultural-ENv2 Llama-SEA-LION-Guard-Cultural-ENv2 EMPTY http://localhost:8001/v1 > inference_safeguard_llama_sealionguard_en_cultural_v2_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating aisingapore/Llama-Guard-v3-N-S..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-Guard-v3-N-S --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-Guard-v3-N-S Llama-Guard-v3-N-S EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_v3_ns_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating aisingapore/Llama-Guard-v3-S2S..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve aisingapore/Llama-Guard-v3-S2S --port 8001 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-Guard-v3-S2S Llama-Guard-v3-S2S EMPTY http://localhost:8001/v1 > inference_safeguard_llama_sealionguard_v3_s2s_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating aisingapore/Llama-Guard-v3-S2U..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-Guard-v3-S2U --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-Guard-v3-S2U Llama-Guard-v3-S2U EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_v3_s2u_api_vllm_0.8.5.post1.out

# 1024 (DONE)
echo "Evaluating aisingapore/Llama-Guard-v3-Half..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve aisingapore/Llama-Guard-v3-Half --port 8000 --download_dir ./data/models --tensor_parallel_size 4
sh api_inference_safeguard.sh SealionGuard aisingapore/Llama-Guard-v3-Half Llama-Guard-v3-Half EMPTY http://localhost:8000/v1 > inference_safeguard_llama_sealionguard_v3_half_api_vllm_0.8.5.post1.out


echo "Evaluating SealionGuardAPI..."
sh api_inference_safeguard.sh SealionGuardAPI aisingapore/Llama-SEA-LION-Guard SealionGuardAPI sk-NgDAhFUD0Y2PfBCKiOSsYA https://dev.api.sea-lion-inference.com/v1 > inference_safeguard_llama_sealionguard_api_aisg.out