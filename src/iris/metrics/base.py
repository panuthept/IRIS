from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from iris.data_types import ModelResponse, EvaluationResult, SummarizedResult


class Metric(ABC):
    @abstractmethod
    def _compute_scores(self, response: ModelResponse, **kwargs) -> Dict:
        raise NotImplementedError

    def eval(self, response: ModelResponse, **kwargs) -> EvaluationResult:
        result = EvaluationResult.from_response(response)
        result.scores.update(self._compute_scores(response, **kwargs))
        return result
    
    def eval_batch(self, responses: List[ModelResponse], verbose: bool = True, **kwargs) -> Tuple[List[EvaluationResult], SummarizedResult]:
        all_results = [self.eval(response, **kwargs) for response in tqdm(responses, disable=not verbose)]
        summarized_result = SummarizedResult.from_results(all_results)
        return all_results, summarized_result


