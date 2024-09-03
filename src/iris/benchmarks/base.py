from abc import ABC
from iris.data_types import SummarizedResult
from iris.model_wrappers.generative_models import GenerativeLLM


class Benchmark(ABC):
    def evaluate(self, model: GenerativeLLM) -> SummarizedResult:
        raise NotImplementedError