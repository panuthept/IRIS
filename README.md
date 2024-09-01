# IRIS
**I**mproving **R**obustness of LLMs on Input Variations by Mitigating Spurious **I**ntermediate **S**tates.

# Download datasets
```bash
cd data
sh download_datasets.sh
```

# Download models
```bash
cd data
python download_model.py --model_name Qwen/Qwen2-0.5B-Instruct
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
