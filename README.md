# IRIS
**I**mproving **R**obustness of LLMs on Input Variations by Mitigating Spurious **I**ntermediate **S**tates.

# Download datasets
```bash
cd data
sh download_datasets.sh
```

# Usage
```python
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from iris.metrics.consistency import ConsistencyRateMetric
from iris.data_types import Sample, ModelResponse, EvaluationResult
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


dataset = JailbreakBenchDataset(attack_engine="PAIR")
samples: List[Sample] = dataset.as_samples(split="harmful")

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
responses: List[ModelResponse] = model.complete_batch(samples)

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
