from typing import List, Dict
from iris.metrics.base import Metric
from iris.data_types import ModelResponse


class ExactMatchMetric(Metric):
    def _compute_score(self, text: str, references: List[str]) -> float:
        return max([float(text == ref) for ref in references])
    
    def _compute_prompt_clf_scores(self, instructions_pred_label, instructions_true_label) -> Dict:
        return {"exact_match": [self._compute_score(pred_label, [true_label]) for pred_label, true_label in zip(instructions_pred_label, instructions_true_label)]}
    
    def _compute_response_clf_scores(self, query, instructions, answers_pred_label, answers_true_label) -> Dict:
        return {"exact_match": [self._compute_score(pred_label, [true_label]) for pred_label, true_label in zip(answers_pred_label, answers_true_label)]}
    
    def _compute_answers_scores(self, query, instructions, answers, reference_answers) -> Dict:
        return {"exact_match": [self._compute_score(answer, reference_answers) for answer in answers]}


if __name__ == "__main__":
    from iris.data_types import Sample, ModelResponse

    metric = ExactMatchMetric()
    sample = Sample(
        instructions=[
            "Output whether the sentiment of the input sentence is positive or negative.",
            "Given an input text, output whether the sentiment is positive or negative.",
            "For each input, determine if the sentiment in the input is prone to negative or positive opinion.",
            "For each input, determine whether it expresses a positive or a negative opinion.",
        ],
        instructions_true_label=["Benign", "Benign", "Harmful", "Harmful"],
        instructions_pred_label=["Benign", "Harmful", "Benign", "Harmful"],
    )
    response = ModelResponse.from_sample(sample)
    response.answers = ["Positive", "Positive", "Negative", "Negative"]
    response.answers_true_label = ["Positive", "Positive", "Negative", "Negative"]
    response.answers_pred_label = ["Positive", "Negative", "Positive", "Negative"]
    response.reference_answers = ["Positive", "Positive", "Negative", "Negative"]

    # Test eval_prompt_clf
    result = metric.eval_prompt_clf(sample)
    print(result.scores)
    assert result.scores["exact_match"]["all"] == [1.0, 0.0, 0.0, 1.0]
    # Test eval_response_clf
    result = metric.eval_response_clf(response)
    print(result.scores)
    assert result.scores["exact_match"]["all"] == [1.0, 0.0, 0.0, 1.0]
    # Test eval_answers
    result = metric.eval_answers(response)
    print(result.scores)
    assert result.scores["exact_match"]["all"] == [1.0, 1.0, 1.0, 1.0]

    print("Passed test!")