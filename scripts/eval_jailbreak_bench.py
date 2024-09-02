import argparse
from typing import List
from iris.data_types import ModelResponse
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai_like import OpenAILike
from iris.utilities.loaders import load_model_answers
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.metrics.jailbreaking import RefusalRateMetric, SafeResponseRateMetric


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    args = parser.parse_args()

    output_paths = [
        f"./outputs/JailbreakBenchDataset/None/benign/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchDataset/None/harmful/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchDataset/GCG/harmful/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchDataset/JBC/harmful/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchDataset/PAIR/harmful/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchDataset/prompt_with_random_search/harmful/{args.model_name}/response.jsonl",
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
        all_results, summarized_result = metric.eval_batch(responses, verbose=False)
        print(f"Refusal rate: {summarized_result.scores}")

        metric = SafeResponseRateMetric(
            judge=APIGenerativeLLM(
                llm=OpenAILike(
                    model="meta-llama/Llama-Guard-3-8B",
                    api_key="EMPTY",
                    api_base="http://10.204.100.70:11700/v1",
                ),
                cache_path="./cache",
            )
        )
        all_results, summarized_result = metric.eval_batch(responses, verbose=False)
        print(f"Safe response rate: {summarized_result.scores}")
        print("-" * 100)