import os
from tqdm import tqdm
from typing import List, Dict
from iris.benchmarks.base import Benchmark
from iris.metrics import Metric, ExactMatchMetric
from iris.data_types import SummarizedResult
from iris.prompt_template import PromptTemplate
from iris.data_types import Sample, ModelResponse
from iris.datasets import PromptSourceDataset
from iris.model_wrappers.generative_models import GenerativeLLM
from iris.utilities.loaders import save_responses, load_responses


class PromptSourceBenchmark(Benchmark):
    def __init__(
        self, 
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/PromptSourceBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template, 
            save_path=save_path, 
        )
        self.task_name_map = {
            "anli_r1": "ANLI R1",
            "anli_r2": "ANLI R2",
            "anli_r3": "ANLI R3",
            "cb": "CB",
            "rte": "RTE",
            "copa" :"COPA",
            "hellaswag": "HellaSwag",
            # "story_cloze": "Story Cloze",
            "wsc": "WSC",
            "winogrande": "Winogrande",
            "wic": "WiC",
        }

    def get_metrics(self) -> List[Metric]:
        return [ExactMatchMetric()]

    def _rename_task(self, task: str) -> str:
        return self.task_name_map[task]
    
    def task_available(self) -> List[str]:
        return PromptSourceDataset.task_available()

    def evaluate(
        self, 
        model: GenerativeLLM = None, 
        model_name: str = None,
        tasks: List[str] = None,
        prompt_name: List[str] = None,
    ) -> Dict[str, SummarizedResult]:
        if model is None:
            assert model_name is not None, "Either model or model_name must be provided"
        model_name = model.get_model_name() if model is not None else model_name
      
        if tasks is None:
            tasks = self.task_available()
        tasks = [task for task in tasks if task in self.task_name_map]
        
        # Inference for each task
        for task in tqdm(tasks, desc="Inference"): 
            output_path = f"{self.save_path}/{task}/{model_name}"
            if os.path.exists(f"{output_path}/response.jsonl"):
                continue
            # Load the dataset
            dataset = PromptSourceDataset(task_name=task)
            samples: List[Sample] = dataset.as_samples(split="test", prompt_template=self.prompt_template)
            # Get the responses
            responses: List[ModelResponse] = model.complete_batch(samples)
            # Save the responses
            os.makedirs(output_path, exist_ok=True)
            save_responses(responses, f"{output_path}/response.jsonl")
          
        # Evaluate the responses
        benchmark_results = {}
        for task in tqdm(tasks, desc="Evaluation"):   
            output_path = f"{self.save_path}/{task}/{model_name}"
            # Load responses
            responses: List[ModelResponse] = load_responses(f"{output_path}/response.jsonl")

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

    benchmark = PromptSourceBenchmark()
    
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