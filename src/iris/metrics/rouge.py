import numpy as np
from typing import Dict, List
from rouge_score import rouge_scorer
from iris.metrics.base import Metric
from iris.data_types import ModelResponse


class RougeMetric(Metric):
    def __init__(self, rouge_type: str = "rougeL", use_stemmer: bool = True):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)
    
    def _compute_prompt_clf_scores(self, instructions_pred_label, instructions_true_label) -> Dict[str, List]:
        results = {f"{self.rouge_type}-precision": [], f"{self.rouge_type}-recall": [], f"{self.rouge_type}-fmeasure": []}
        for pred_label, true_label in zip(instructions_pred_label, instructions_true_label):
            score = self.scorer.score_multi([true_label], pred_label)[self.rouge_type]
            results[f"{self.rouge_type}-precision"].append(score.precision)
            results[f"{self.rouge_type}-recall"].append(score.recall)
            results[f"{self.rouge_type}-fmeasure"].append(score.fmeasure)
        return results
    
    def _compute_response_clf_scores(self, query, instructions, answers_pred_label, answers_true_label) -> Dict[str, List]:
        results = {f"{self.rouge_type}-precision": [], f"{self.rouge_type}-recall": [], f"{self.rouge_type}-fmeasure": []}
        for pred_label, true_label in zip(answers_pred_label, answers_true_label):
            score = self.scorer.score_multi([true_label], pred_label)[self.rouge_type]
            results[f"{self.rouge_type}-precision"].append(score.precision)
            results[f"{self.rouge_type}-recall"].append(score.recall)
            results[f"{self.rouge_type}-fmeasure"].append(score.fmeasure)
        return results
    
    def _compute_answers_scores(self, query, instructions, answers, reference_answers) -> Dict[str, List]:
        results = {f"{self.rouge_type}-precision": [], f"{self.rouge_type}-recall": [], f"{self.rouge_type}-fmeasure": []}
        for answer in answers:
            score = self.scorer.score_multi(reference_answers, answer)[self.rouge_type]
            results[f"{self.rouge_type}-precision"].append(score.precision)
            results[f"{self.rouge_type}-recall"].append(score.recall)
            results[f"{self.rouge_type}-fmeasure"].append(score.fmeasure)
        return results


if __name__ == "__main__":
    from iris.data_types import Sample, ModelResponse

    metric = RougeMetric()
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
    assert result.scores["rougeL-fmeasure"]["all"] == [1.0, 0.0, 0.0, 1.0]
    # Test eval_response_clf
    result = metric.eval_response_clf(response)
    print(result.scores)
    assert result.scores["rougeL-fmeasure"]["all"] == [1.0, 0.0, 0.0, 1.0]
    # Test eval_answers
    result = metric.eval_answers(response)
    print(result.scores)
    assert result.scores["rougeL-fmeasure"]["all"] == [1.0, 1.0, 1.0, 1.0]

    print("Passed test!")


    # response = ModelResponse(
    #     instructions=[
    #         "You are given a sentence in Persian. Your job is to translate the Farsi sentence into Polish.",
    #     ],
    #     query="ایلان: درسته. این مهمه که بخش های راکت بتونند برگردند و بتونند به سایت پرتاب برگردند و اماده پرتاب باشند در عرض چند ساعت.",
    #     reference_answers=["EM: Owszem. Ważne jest by rakieta mogła wrócić do lądowiska i wystartować ponownie w ciągu kilku minut."],
    #     answers=[
    #         "Elon: To prawda. Ważne jest, aby części rakiety mogły wrócić i mogły wrócić na miejsce startu oraz były gotowe do startu w ciągu kilku godzin.",
    #     ],
    # )

    # metric = RougeMetric()
    # result = metric.eval(response)
    # print(result)
    # assert result.scores["precision"]["mean"] == 1/3
    # assert result.scores["recall"]["mean"] == 0.5
    # assert result.scores["fmeasure"]["mean"] == 0.4
    # assert result.scores["precision"]["std"] == 0.0
    # assert result.scores["recall"]["std"] == 0.0
    # assert result.scores["fmeasure"]["std"] == 0.0

    # metric = RougeMetric(rouge_type="rouge1")
    # result = metric.eval(response)
    # print(result)
    # assert result.scores["precision"]["mean"] == 0.3939393939393939
    # assert result.scores["recall"]["mean"] == 0.5909090909090909
    # assert result.scores["fmeasure"]["mean"] == 0.4727272727272727
    # assert result.scores["precision"]["std"] == 0.0
    # assert result.scores["recall"]["std"] == 0.0
    # assert result.scores["fmeasure"]["std"] == 0.0
    # print("Passed test!")