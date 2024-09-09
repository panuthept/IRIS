import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from iris.datasets import JailbreakDataset
from iris.data_types import SummarizedResult
from iris.prompt_template import PromptTemplate
from iris.data_types import Sample, ModelResponse
from llama_index.llms.together import TogetherLLM
from llama_index.llms.openai_like import OpenAILike
from iris.utilities.loaders import save_model_answers, load_model_answers
from iris.model_wrappers.generative_models import GenerativeLLM, APIGenerativeLLM
from iris.metrics import RefusalRateMetric, SafeResponseRateMetric, ExactMatchMetric, Metric


class Benchmark(ABC):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/Benchmark",
    ):
        self.prompt_template = prompt_template
        self.save_path = save_path

    @abstractmethod
    def get_metrics(self) -> List[Metric]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: GenerativeLLM) -> SummarizedResult:
        raise NotImplementedError
    

class JailbreakBenchmark(Benchmark):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreakBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template,
            save_path=save_path
        )

    def get_evaluation_settings(self) -> List[Dict[str, Any]]:
        # Return a list of [{"intention": string, "category": string, "attack_engine": string, "save_name": string, "setting_name": string}, ...]
        raise NotImplementedError

    def get_dataset(self, intention: str, category: str, attack_engine: str) -> JailbreakDataset:
        raise NotImplementedError

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
        inference_only: bool = False
    ) -> Dict[str, SummarizedResult]:
        if model is None:
            assert model_name is not None, "Either model or model_name must be provided"
        model_name = model.get_model_name() if model is not None else model_name

        # Inference for each task
        evaluation_settings = self.get_evaluation_settings()
        for setting in tqdm(evaluation_settings, desc="Inference"):
            output_path = f"{self.save_path}/{setting['save_name']}/{model_name}"
            if os.path.exists(f"{output_path}/response.jsonl"):
                continue
            # Load the dataset
            dataset = self.get_dataset(
                intention=setting.get("intention", None),
                category=setting.get("category", None),
                attack_engine=setting.get("attack_engine", None),
            )
            samples: List[Sample] = dataset.as_samples(split="test", prompt_template=self.prompt_template)
            # Get the responses
            responses: List[ModelResponse] = model.complete_batch(samples)
            # Save the responses
            os.makedirs(output_path, exist_ok=True)
            save_model_answers(responses, f"{output_path}/response.jsonl")

        if inference_only:
            return

        # Evaluate the responses
        benchmark_results = {}
        for setting in tqdm(evaluation_settings, desc="Evaluation"):
            output_path = f"{self.save_path}/{setting['save_name']}/{model_name}"
            # Load responses
            responses: List[ModelResponse] = load_model_answers(f"{output_path}/response.jsonl")

            # Evaluate responses
            results = {}
            for metric in self.get_metrics():
                _, summarized_result = metric.eval_batch(responses, verbose=False)
                results.update(summarized_result.scores)
            benchmark_results[setting["setting_name"]] = results
        return benchmark_results
    

class JailbreakPromptCLFBenchmark(JailbreakBenchmark):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/JailbreakPromptCLFBenchmark",
    ):
        super().__init__(
            prompt_template=prompt_template,
            save_path=save_path
        )

    def get_metrics(self) -> List[Metric]:
        return [ExactMatchMetric()]