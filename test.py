import os
import torch
from typing import List
from iris.data_types import Sample, ModelResponse
from llama_index.llms.together import TogetherLLM
from iris.metrics.jailbreaking import RefusalRateMetric
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.datasets.instruction_induction import InstructionIndutionDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dataset: List[Sample] -> Model: List[ModelResponse] -> Metric: Tuple[List[EvaluationResult], SummarizedResult]
    print("Test JailbreakBench!!")
    dataset = JailbreakBenchDataset(
        attack_engine="PAIR", 
        cache_dir="./data/datasets/jailbreak_bench",
    )
    samples: List[Sample] = dataset.as_samples(split="harmful")[:1]
    print(samples[0].get_prompts()[0])

    model = HuggfaceGenerativeLLM(
        "Qwen/Qwen2-0.5B-Instruct",
        max_tokens=512,
        pipeline_kwargs={
            "torch_dtype": torch.bfloat16,
            "model_kwargs": {
                "cache_dir": "./data/models",
                "local_files_only": False,
            }
        },
        cache_path="./cache",
    )
    responses: List[ModelResponse] = model.complete_batch(samples)
    print(f"Responses: {responses[0].answers}")

    metric = RefusalRateMetric(
        judge=APIGenerativeLLM(
            llm=TogetherLLM(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f"
            ),
            cache_path="./cache",
        )
    )
    all_results, summarized_result = metric.eval_batch(responses)
    print(summarized_result.scores)


    print("\n\nTest Instruction Induction!!")
    dataset = InstructionIndutionDataset(
        induction_engine="sentiment", 
        cache_dir="./data/datasets/instruction_induction/data",
    )
    samples: List[Sample] = dataset.as_samples()[:1]
    print(samples[0].get_prompts()[0])

    model = HuggfaceGenerativeLLM(
        "Qwen/Qwen2-0.5B-Instruct",
        max_tokens=512,
        pipeline_kwargs={
            "torch_dtype": torch.bfloat16,
            "model_kwargs": {
                "cache_dir": "./data/models",
                "local_files_only": False,
            }
        },
        cache_path="./cache",
    )
    responses: List[ModelResponse] = model.complete_batch(samples)
    print(f"Responses: {responses[0].answers}")

    metric = RefusalRateMetric(
        judge=APIGenerativeLLM(
            llm=TogetherLLM(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f"
            ),
            cache_path="./cache",
        )
    )
    all_results, summarized_result = metric.eval_batch(responses)
    print(summarized_result.scores)