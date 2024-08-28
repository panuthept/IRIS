from typing import List, Dict
from abc import ABC, abstractmethod
from iris.data_types import ModelResponse, EvaluationResult


class Metric(ABC):
    @abstractmethod
    def _compute_scores(self, response: ModelResponse) -> Dict:
        raise NotImplementedError

    def eval(self, response: ModelResponse) -> EvaluationResult:
        result = EvaluationResult.from_response(response)
        result.scores.update(self._compute_scores(response))
        return result
    
    def eval_batch(self, responses: List[ModelResponse]) -> List[EvaluationResult]:
        return [self.eval(response) for response in responses]


