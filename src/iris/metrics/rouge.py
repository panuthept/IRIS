import numpy as np
from typing import Dict
from rouge_score import rouge_scorer
from iris.metrics.base import Metric
from iris.data_types import ModelResponse


class RougeMetric(Metric):
    def __init__(self, rouge_type: str = "rougeL", use_stemmer: bool = True):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)

    def _compute_scores(self, response: ModelResponse) -> Dict:
        scores = {"precision": [], "recall": [], "fmeasure": []}
        for answer in response.answers:
            max_score = {"precision": 0, "recall": 0, "fmeasure": 0}
            for ref in response.reference_answers:
                score = self.scorer.score(ref, answer)[self.rouge_type]
                if score.fmeasure > max_score["fmeasure"]:
                    max_score = {
                        "precision": score.precision,
                        "recall": score.recall,
                        "fmeasure": score.fmeasure,
                    }
            for key in scores:
                scores[key].append(max_score[key])
        mean_scores = {key: np.mean(scores[key]) for key in scores}
        std_scores = {key: np.std(scores[key]) for key in scores}
        return {
            "precision": {"mean": mean_scores["precision"], "std": std_scores["precision"], "all": scores["precision"]},
            "recall": {"mean": mean_scores["recall"], "std": std_scores["recall"], "all": scores["recall"]},
            "fmeasure": {"mean": mean_scores["fmeasure"], "std": std_scores["fmeasure"], "all": scores["fmeasure"]},
        }


if __name__ == "__main__":
    response = ModelResponse(
        instructions=[
            "You are given a sentence in Persian. Your job is to translate the Farsi sentence into Polish.",
        ],
        query="ایلان: درسته. این مهمه که بخش های راکت بتونند برگردند و بتونند به سایت پرتاب برگردند و اماده پرتاب باشند در عرض چند ساعت.",
        reference_answers=["EM: Owszem. Ważne jest by rakieta mogła wrócić do lądowiska i wystartować ponownie w ciągu kilku minut."],
        answers=[
            "Elon: To prawda. Ważne jest, aby części rakiety mogły wrócić i mogły wrócić na miejsce startu oraz były gotowe do startu w ciągu kilku godzin.",
        ],
    )

    metric = RougeMetric()
    result = metric.eval(response)
    print(result)
    assert result.scores["precision"]["mean"] == 1/3
    assert result.scores["recall"]["mean"] == 0.5
    assert result.scores["fmeasure"]["mean"] == 0.4
    assert result.scores["precision"]["std"] == 0.0
    assert result.scores["recall"]["std"] == 0.0
    assert result.scores["fmeasure"]["std"] == 0.0

    metric = RougeMetric(rouge_type="rouge1")
    result = metric.eval(response)
    print(result)
    assert result.scores["precision"]["mean"] == 0.3939393939393939
    assert result.scores["recall"]["mean"] == 0.5909090909090909
    assert result.scores["fmeasure"]["mean"] == 0.4727272727272727
    assert result.scores["precision"]["std"] == 0.0
    assert result.scores["recall"]["std"] == 0.0
    assert result.scores["fmeasure"]["std"] == 0.0
    print("Passed test!")