import argparse
from typing import List
from iris.data_types import ModelResponse
from iris.utilities.loaders import load_responses
from iris.metrics.exact_match import ExactMatchMetric


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    args = parser.parse_args()

    output_paths = [
        f"./outputs/JailbreakBenchResponseCLFDataset/benign/{args.model_name}/response.jsonl",
        f"./outputs/JailbreakBenchResponseCLFDataset/harmful/{args.model_name}/response.jsonl",
    ]

    for output_path in output_paths:
        responses: List[ModelResponse] = load_responses(output_path)
        print(f"Loaded {len(responses)} responses from {output_path}")

        metric = ExactMatchMetric()
        all_results, summarized_result = metric.eval_batch(responses, verbose=False)
        print(f"EM: {summarized_result.scores}")