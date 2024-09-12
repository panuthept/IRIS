# IRIS
**I**mproving **R**obustness of LLMs on Input Variations by Mitigating Spurious **I**ntermediate **S**tates.

**Disclaimer: This is still a developing work, there is no useful artifact yet.**

# Setup
```bash
conda create -n iris python==3.11.4
conda activate iris

# Select the appropriate PyTorch version based on your CUDA version
# CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch
```
```bash
git clone https://github.com/panuthept/IRIS.git
cd IRIS
pip install -e .
```

# Download datasets
```bash
cd IRIS/data
sh download_datasets.sh
```

# Download models
```bash
cd IRIS/data
python download_model.py --model_name <HUGGINGFACE_MODEL_NAME> # e.g., Qwen/Qwen2-0.5B-Instruct
```

# Usage
```python
from llama_index.llms.together import TogetherLLM
from iris.metrics.jailbreaking import RefusalRateMetric
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


dataset = JailbreakBenchDataset(attack_engine="PAIR")
samples = dataset.as_samples(split="harmful")

model = HuggfaceGenerativeLLM(
    "Qwen/Qwen2-0.5B-Instruct",
    max_tokens=512,
    pipeline_kwargs={
        "torch_dtype": torch.bfloat16,
        "model_kwargs": {
            "cache_dir": "./models",
            "local_files_only": False,
        }
    },
    cache_path="./cache",
)
responses = model.complete_batch(samples)

metric = RefusalRateMetric(
    judge=APIGenerativeLLM(
        llm=TogetherLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key="<YOUR_API_KEY>"
        ),
        cache_path="./cache",
    )
)
all_results, summarized_result = metric.eval_batch(responses)
```
