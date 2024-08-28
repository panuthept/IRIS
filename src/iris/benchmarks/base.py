from typing import List
from abc import ABC, abstractmethod
from iris.data_types import Sample, ModelResponse, EvaluationResult


class Benchmark(ABC):
    def get_train_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    def get_test_set(self, *args, **kwargs) -> List[Sample]:
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, responses: ModelResponse) -> EvaluationResult:
        raise NotImplementedError
    
    def evaluate(self, responses: List[ModelResponse]) -> List[EvaluationResult]:
        return [self._evaluate(response) for response in responses]