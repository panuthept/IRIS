import os
from tqdm import tqdm
from typing import List, Dict
from iris.benchmarks.base import Benchmark
from iris.data_types import SummarizedResult
from iris.datasets import JailbreakBenchDataset
from iris.prompt_template import PromptTemplate
from iris.data_types import Sample, ModelResponse
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai_like import OpenAILike
from iris.model_wrappers.generative_models import GenerativeLLM
from iris.model_wrappers.generative_models import APIGenerativeLLM
from iris.utilities.loaders import save_model_answers, load_model_answers
from iris.metrics import RefusalRateMetric, SafeResponseRateMetric, Metric


class JailbreakBenchBenchmark(Benchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreakBenchBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template, 
            save_path=save_path
        )

    def get_metrics(self) -> List[Metric]:
        return [
            RefusalRateMetric(
                judge=APIGenerativeLLM(
                    llm=TogetherLLM(
                        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                        api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f"
                    ),
                    cache_path="./cache",
                )
            ),
            SafeResponseRateMetric(
                judge=APIGenerativeLLM(
                    llm=OpenAILike(
                        model="meta-llama/Llama-Guard-3-8B",
                        api_key="EMPTY",
                        api_base="http://10.204.100.70:11700/v1",
                    ),
                    cache_path="./cache",
                )
            )
        ]

    def evaluate(
        self, 
        model: GenerativeLLM = None, 
        model_name: str = None,
    ) -> Dict[str, SummarizedResult]:
        if model is None:
            assert model_name is not None, "Either model or model_name must be provided"
        model_name = model.get_model_name() if model is not None else model_name

        tasks = [
            "None/benign",
            "None/harmful",
            "GCG/harmful",
            "JBC/harmful",
            "PAIR/harmful",
            "prompt_with_random_search/harmful",
        ]

        # Inference for each task
        for task in tqdm(tasks, desc="Inference"):
            output_path = f"{self.save_path}/{task}/{model_name}"
            if os.path.exists(f"{output_path}/response.jsonl"):
                continue
            # Load the dataset
            attack_engine, split = task.split("/")
            dataset = JailbreakBenchDataset(
                attack_engine=attack_engine,
            )
            samples: List[Sample] = dataset.as_samples(split=split, prompt_template=self.prompt_template)
            # Get the responses
            responses: List[ModelResponse] = model.complete_batch(samples)
            # Save the responses
            os.makedirs(output_path, exist_ok=True)
            save_model_answers(responses, f"{output_path}/response.jsonl")

        # Evaluate the responses
        benchmark_results = {}
        for task in tqdm(tasks, desc="Evaluation"):
            output_path = f"{self.save_path}/{task}/{model_name}"
            # Load responses
            responses: List[ModelResponse] = load_model_answers(f"{output_path}/response.jsonl")

            # Evaluate responses
            task_results = {}
            for metric in self.get_metrics():
                _, summarized_result = metric.eval_batch(responses, verbose=False)
                task_results.update(summarized_result.scores)
            benchmark_results[self._rename_task(task)] = task_results
        return benchmark_results


if __name__ == "__main__":
    import torch
    from iris.model_wrappers.generative_models import HuggfaceGenerativeLLM

    benchmark = JailbreakBenchBenchmark()

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