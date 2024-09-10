import numpy as np
from abc import ABC
from tqdm import tqdm
from deprecated import deprecated
from collections import defaultdict
from typing import List, Dict, Tuple
from iris.data_types import Sample, ModelResponse, EvaluationResult, SummarizedResult


class Metric(ABC):       
    def _compute_prompt_clf_scores(self, instructions_pred_label, instructions_true_label) -> Dict[str, List]:
        raise NotImplementedError
    
    def _compute_response_clf_scores(self, query, instructions, answers_pred_label, answers_true_label) -> Dict[str, List]:
        raise NotImplementedError
    
    def _compute_answers_scores(self, query, instructions, answers, reference_answers) -> Dict[str, List]:
        raise NotImplementedError
    
    def compute_scores(
        self, 
        query: str = None,
        instructions: List[str] = None, 
        instructions_pred_label: List[str] = None,
        instructions_true_label: List[str] = None,
        answers: List[str] = None, 
        answers_pred_label: List[str] = None,
        answers_true_label: List[str] = None,
        reference_answers: List[str] = None, 
    ) -> Dict:
        """ 
        Compute mean and standard deviation of the metric scores.
        Return a dictionary of:
        {
            "metric_name": {"mean": float, "std": float, "all": List[float]}
            "metric_name": {"mean": float, "std": float, "all": List[float]}
        }
        """
        if instructions_true_label:
            scores = self._compute_prompt_clf_scores(instructions_pred_label, instructions_true_label)
        elif answers_true_label:
            scores = self._compute_response_clf_scores(query, instructions, answers_pred_label, answers_true_label)
        elif reference_answers:
            scores = self._compute_answers_scores(query, instructions, answers, reference_answers)
        else:
            raise ValueError("Invalid input")

        mean_scores = defaultdict(lambda: defaultdict(list))
        for metric_name, scores in scores.items():
            mean_scores[metric_name]["all"] = scores
            mean_scores[metric_name]["mean"] = np.mean(scores)
            mean_scores[metric_name]["std"] = np.std(scores)
        return mean_scores
    
    def eval_prompt_clf(self, sample: Sample) -> EvaluationResult:
        result = EvaluationResult.from_sample(sample)
        result.scores.update(self.compute_scores(
            instructions_pred_label=sample.instructions_pred_label, 
            instructions_true_label=sample.instructions_true_label, 
        ))
        return result
    
    def eval_response_clf(self, response: ModelResponse) -> EvaluationResult:
        result = EvaluationResult.from_response(response)
        result.scores.update(self.compute_scores(
            query=response.query,
            instructions=response.instructions, 
            answers_pred_label=response.answers_pred_label, 
            answers_true_label=response.answers_true_label,
        ))
        return result

    def eval_answers(self, response: ModelResponse) -> EvaluationResult:
        result = EvaluationResult.from_response(response)
        result.scores.update(self.compute_scores(
            query=response.query,
            instructions=response.instructions, 
            answers=response.answers, 
            reference_answers=response.reference_answers, 
        ))
        return result

    def eval_prompt_clf_batch(self, samples: List[Sample], verbose: bool = True, **kwargs) -> Tuple[List[EvaluationResult], SummarizedResult]:
        all_results = [self.eval_prompt_clf(sample) for sample in tqdm(samples, disable=not verbose)]
        summarized_result = SummarizedResult.from_results(all_results)
        return all_results, summarized_result
    
    def eval_response_clf_batch(self, responses: List[ModelResponse], verbose: bool = True, **kwargs) -> Tuple[List[EvaluationResult], SummarizedResult]:
        all_results = [self.eval_response_clf(response) for response in tqdm(responses, disable=not verbose)]
        summarized_result = SummarizedResult.from_results(all_results)
        return all_results, summarized_result

    def eval_answers_batch(self, responses: List[ModelResponse], verbose: bool = True, **kwargs) -> Tuple[List[EvaluationResult], SummarizedResult]:
        all_results = [self.eval_answers(response) for response in tqdm(responses, disable=not verbose)]
        summarized_result = SummarizedResult.from_results(all_results)
        return all_results, summarized_result

    @deprecated
    def eval(self, response: ModelResponse, **kwargs) -> EvaluationResult:
        return self.eval_answers(response)
    
    @deprecated
    def eval_batch(self, responses: List[ModelResponse], verbose: bool = True, **kwargs) -> Tuple[List[EvaluationResult], SummarizedResult]:
        return self.eval_answers_batch(responses, verbose=verbose, **kwargs)