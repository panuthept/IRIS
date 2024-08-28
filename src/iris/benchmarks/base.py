from typing import List
from abc import ABC, abstractmethod
from iris.data_types import Sample, GenerativeLLMResponse, GenerativeLLMResult


class Benchmark(ABC):
    def get_train_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    def get_test_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, responses: List[GenerativeLLMResponse]) -> List[GenerativeLLMResult]:
        raise NotImplementedError