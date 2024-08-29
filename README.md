# IRIS
**I**mproving **R**obustness of LLMs on Input Variations by Mitigating Spurious **I**ntermediate **S**tates.

# Usage
```python
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from iris.metrics.consistency import ConsistencyRateMetric
from iris.data_types import Sample, ModelResponse, EvaluationResult
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


samples: List[Sample] = [
    Sample(
        instructions=[
            "Output whether the sentiment of the input sentence is positive or negative.",
            "Given an input text, output whether the sentiment is positive or negative.",
            "For each input, determine if the sentiment in the input is prone to negative or positive opinion.",
            "For each input, determine whether it expresses a positive or a negative opinion.",
            "Classify the sentiment of the input sentence (options are positive or negative)",
            "write \"positive\" if the input is a positive review, and \"negative\" if the input is a negative review",
            "Determine whether the sentiment is positive or negative",
            "Output whether the sentiment is positive or negative"
        ],
        query="A tender, heartfelt family drama.",
        reference_answers=["Positive"],
    )
]

model = APIGenerativeLLM(
    llm=TogetherLLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key="<YOUR_API_KEY>",
    ),
    system_prompt="You will be given a instruction and query, use deterministic output as 'Positive' or 'Negative' without additional information or character.",
    post_processing=lambda x: x.strip().capitalize(),
)
responses: List[ModelResponse] = model.complete_batch(samples)

metric = ConsistencyRateMetric(
    llm=OpenAI(
        model="gpt-4o", 
        api_key="<YOUR_API_KEY>",
    ),
)
results: List[EvaluationResult] = metric.eval_batch(responses, reference_index = 0)
```
