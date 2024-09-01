from typing import List
from iris.data_types import ModelResponse
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from iris.utilities.loaders import load_model_answers
from iris.metrics.jailbreaking import RefusalRateMetric
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


if __name__ == "__main__":
    output_paths = [
        "./outputs/JailbreakBenchDataset/None/benign/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
        "./outputs/JailbreakBenchDataset/None/harmful/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
        "./outputs/JailbreakBenchDataset/GCG/harmful/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
        "./outputs/JailbreakBenchDataset/JBC/harmful/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
        "./outputs/JailbreakBenchDataset/PAIR/harmful/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
        "./outputs/JailbreakBenchDataset/prompt_with_random_search/harmful/Qwen/Qwen2-0.5B-Instruct/response.jsonl",
    ]

    for output_path in output_paths:
        responses: List[ModelResponse] = load_model_answers(output_path)
        print(f"Loaded {len(responses)} responses from {output_path}")
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