import numpy as np
from typing import Dict
from iris.metrics.base import Metric
from iris.data_types import ModelResponse


class ExactMatchMetric(Metric):
    def _compute_scores(self, response: ModelResponse) -> Dict:
        scores = []
        for answer in response.answers:
            max_score = max([float(answer == ref) for ref in response.reference_answers])
            scores.append(max_score)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        return {"exact_match": {"mean": mean_score, "std": std_score}}


if __name__ == "__main__":
    response = ModelResponse(
        instructions=[
            "Output whether the sentiment of the input sentence is positive or negative.",
            "Given an input text, output whether the sentiment is positive or negative.",
            "For each input, determine if the sentiment in the input is prone to negative or positive opinion.",
            "For each input, determine whether it expresses a positive or a negative opinion.",
            "Classify the sentiment of the input sentence (options are positive or negative)",
            "write \"positive\" if the input is a positive review, and \"negative\" if the input is a negative review",
            "Determine whether the sentiment is positive or negative",
            "Output whether the sentiment is positive or negative",
        ],
        query="A tender, heartfelt family drama.",
        reference_answers=["Positive"],
        answers=[
            "Positive",
            "Negative",
            "Positive",
            "Negative",
            "Positive",
            "Negative",
            "Positive",
            "Negative",
        ],
    )

    metric = ExactMatchMetric()
    result = metric.eval(response)
    print(result)
    assert result.scores["exact_match"]["mean"] == 0.5
    assert result.scores["exact_match"]["std"] == 0.5
    print("Passed test!")