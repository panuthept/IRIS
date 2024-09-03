import os
from tqdm import tqdm
from typing import List, Dict
from iris.benchmarks.base import Benchmark
from iris.data_types import SummarizedResult
from iris.data_types import Sample, ModelResponse
from iris.datasets import InstructionIndutionDataset
from iris.metrics import ExactMatchMetric, RougeMetric
from iris.model_wrappers.generative_models import GenerativeLLM
from iris.utilities.loaders import save_model_answers, load_model_answers


class InstructionIndutionBenchmark(Benchmark):
    def __init__(self, save_path: str = f"./outputs/InstructionIndutionBenchmark"):
        self.save_path = save_path
        self.metrics = [
            ExactMatchMetric(),
            RougeMetric("rougeL"),
        ]

    def evaluate(self, model: GenerativeLLM, tasks: List[str] = None) -> Dict[str, SummarizedResult]:
        if tasks is None:
            tasks = InstructionIndutionDataset.task_available()

        # Inference for each task
        for task in tqdm(tasks, desc="Inference"):
            output_path = f"{self.save_path}/{task}/{model.get_model_name()}"
            if os.path.exists(f"{output_path}/response.jsonl"):
                continue
            # Load the dataset
            dataset = InstructionIndutionDataset(
                task_name=task,
            )
            samples: List[Sample] = dataset.as_samples()
            # Get the responses
            responses: List[ModelResponse] = model.complete_batch(samples)
            # Save the responses
            os.makedirs(output_path, exist_ok=True)
            save_model_answers(responses, f"{output_path}/response.jsonl")

        # Evaluate the responses
        benchmark_results = {}
        for task in tqdm(tasks, desc="Evaluation"):
            output_path = f"{self.save_path}/{task}/{model.get_model_name()}"
            # Load responses
            responses: List[ModelResponse] = load_model_answers(output_path)

            # Evaluate responses
            task_results = {}
            for metric in self.metrics:
                _, summarized_result = metric.eval_batch(responses, verbose=False)
                task_results.update(summarized_result.scores)
            benchmark_results[task] = task_results
        return benchmark_results


if __name__ == "__main__":
    import torch
    from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM

    benchmark = InstructionIndutionBenchmark()

    print(f"CUDA available: {torch.cuda.is_available()}")
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
    print(f"Device: {model.llm.device}")

    results = benchmark.evaluate(model)
    print(results)