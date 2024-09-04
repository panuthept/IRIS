from typing import List
from iris.metrics import Metric
from abc import ABC, abstractmethod
from iris.data_types import SummarizedResult
from iris.prompt_template import PromptTemplate
from iris.model_wrappers.generative_models import GenerativeLLM


class Benchmark(ABC):
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        save_path: str = f"./outputs/InstructionIndutionBenchmark",
    ):
        self.prompt_template = prompt_template
        self.save_path = save_path

    @abstractmethod
    def get_metrics(self) -> List[Metric]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: GenerativeLLM) -> SummarizedResult:
        raise NotImplementedError