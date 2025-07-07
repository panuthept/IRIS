# vLLM version: 0.8.5.post1
# huggingface-cli login --token $1 --add-to-git-credential

# vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 8
# vllm serve meta-llama/Llama-Guard-4-12B --port 8000 --download_dir ./data/models --tensor_parallel_size 8

export TRANSFORMERS_CACHE="/shared/panuthep/cache/huggingface/transformers"
export HF_DATASETS_CACHE="/shared/panuthep/cache/huggingface/datasets"
export HF_HOME="/shared/panuthep/cache/huggingface"
export VLLM_CACHE_ROOT="/shared/panuthep/cache/vllm"

# 1024 (DONE)
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-2-9b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 1
# 1024 Running [3]
sh api_inference_llm_en.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [4]
sh api_inference_llm_in.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [5]
sh api_inference_llm_ms.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [6]
sh api_inference_llm_my.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [7]
sh api_inference_llm_ta.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [8]
sh api_inference_llm_th.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [9]
sh api_inference_llm_tl.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024 Running [10]
sh api_inference_llm_vi.sh google/gemma-2-9b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it gemma-2-9b-it EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh gemma-2-9b-it

# 1023 (DONE)
vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 8
# 1024 Running [3]
sh api_inference_llm_en.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [4]
sh api_inference_llm_in.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [5]
sh api_inference_llm_ms.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [6]
sh api_inference_llm_my.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [7]
sh api_inference_llm_ta.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [8]
sh api_inference_llm_th.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [9]
sh api_inference_llm_tl.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024 Running [10]
sh api_inference_llm_vi.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it gemma-3-27b-it EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh gemma-3-27b-it

# 1024 (DONE)
vllm serve Meta-Llama-3.1-70B-Instruct --port 8000 --tensor_parallel_size 8
# 1024 Running [3]
sh api_inference_llm_en.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [4]
sh api_inference_llm_in.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [5]
sh api_inference_llm_ms.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [6]
sh api_inference_llm_my.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [7]
sh api_inference_llm_ta.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [8]
sh api_inference_llm_th.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [9]
sh api_inference_llm_tl.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024 Running [10]
sh api_inference_llm_vi.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh Meta-Llama-3.1-70B-Instruct

# 1024 (DONE)
vllm serve Llama-3.3-70B-Instruct --port 8000 --tensor_parallel_size 8
# 1023 Running [3]
sh api_inference_llm_en.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [4]
sh api_inference_llm_in.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [5]
sh api_inference_llm_ms.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [6]
sh api_inference_llm_my.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [7]
sh api_inference_llm_ta.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [8]
sh api_inference_llm_th.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [9]
sh api_inference_llm_tl.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1023 Running [10]
sh api_inference_llm_vi.sh Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it Llama-3.3-70B-Instruct EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh Llama-3.3-70B-Instruct

# 1024 (DONE)
CUDA_VISIBLE_DEVICES=1 vllm serve aisingapore/Gemma-SEA-LION-v3-9B-IT --port 8001 --download_dir ./data/models --tensor_parallel_size 1
# 1024 Running [3]
sh api_inference_llm_en.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [4]
sh api_inference_llm_in.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [5]
sh api_inference_llm_ms.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [6]
sh api_inference_llm_my.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [7]
sh api_inference_llm_ta.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [8]
sh api_inference_llm_th.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [9]
sh api_inference_llm_tl.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024 Running [10]
sh api_inference_llm_vi.sh aisingapore/Gemma-SEA-LION-v3-9B-IT Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8001/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it Gemma-SEA-LION-v3-9B-IT EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh Gemma-SEA-LION-v3-9B-IT

# 1023 (DONE)
vllm serve aisingapore/Llama-SEA-LION-v3-70B-IT --port 8000 --download_dir ./data/models --tensor_parallel_size 8
# 1023 Running [3]
sh api_inference_llm_en.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [4]
sh api_inference_llm_in.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [5]
sh api_inference_llm_ms.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [6]
sh api_inference_llm_my.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [7]
sh api_inference_llm_ta.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [8]
sh api_inference_llm_th.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [9]
sh api_inference_llm_tl.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1023 Running [10]
sh api_inference_llm_vi.sh aisingapore/Llama-SEA-LION-v3-70B-IT Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
# 1024
sh api_eval_llm_refusal.sh google/gemma-3-27b-it Llama-SEA-LION-v3-70B-IT EMPTY http://localhost:8000/v1
sh api_eval_llm_safe_response.sh Llama-SEA-LION-v3-70B-IT