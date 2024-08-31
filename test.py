import os
import torch
from typing import List
from iris.data_types import Sample, ModelResponse
from llama_index.llms.together import TogetherLLM
from iris.metrics.jailbreaking import RefusalRateMetric
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dataset: List[Sample] -> Model: List[ModelResponse] -> Metric: Tuple[List[EvaluationResult], SummarizedResult]
    dataset = JailbreakBenchDataset(attack_engine="PAIR")
    samples: List[Sample] = dataset.as_samples(split="harmful")[:1]
    print(samples[0].get_prompts()[0])

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
    # model = APIGenerativeLLM(
    #     llm=TogetherLLM(
    #         # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    #         model="meta-llama/Llama-2-13b-chat-hf",
    #         api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
    #     ),
    # )
    responses: List[ModelResponse] = model.complete_batch(samples)
    print(f"Responses: {responses[0].answers}")

    metric = RefusalRateMetric(
        llm=TogetherLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
        ),
    )
    all_results, summarized_result = metric.eval_batch(responses)
    print(summarized_result.scores)