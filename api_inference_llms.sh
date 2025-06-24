# vLLM version: 0.8.5.post1
# huggingface-cli login --token $1 --add-to-git-credential

# vllm serve google/gemma-3-27b-it --port 8000 --download_dir ./data/models --tensor_parallel_size 8
# vllm serve meta-llama/Llama-Guard-4-12B --port 8000 --download_dir ./data/models --tensor_parallel_size 8

export TRANSFORMERS_CACHE="/shared/panuthep/cache/huggingface/transformers"
export HF_DATASETS_CACHE="/shared/panuthep/cache/huggingface/datasets"
export HF_HOME="/shared/panuthep/cache/huggingface"
export VLLM_CACHE_ROOT="/shared/panuthep/cache/vllm"


vllm serve Meta-Llama-3.1-70B-Instruct --port 8000 --tensor_parallel_size 8
sh api_inference_llm.sh Meta-Llama-3.1-70B-Instruct Meta-Llama-3.1-70B-Instruct EMPTY http://localhost:8000/v1