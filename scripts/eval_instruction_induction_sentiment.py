import argparse
from typing import List
from iris.data_types import ModelResponse
from iris.utilities.loaders import load_model_answers
from iris.metrics.exact_match import ExactMatchMetric


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    args = parser.parse_args()

    output_paths = [
        f"./outputs/InstructionIndutionDataset/sentiment/{args.model_name}/response.jsonl",
    ]
    metric = ExactMatchMetric()

    for output_path in output_paths:
        responses: List[ModelResponse] = load_model_answers(output_path)
        print(f"Loaded {len(responses)} responses from {output_path}")

        results = []
        all_results, summarized_result = metric.eval_batch(responses, verbose=False)
        print(f"EM: {summarized_result.scores}")