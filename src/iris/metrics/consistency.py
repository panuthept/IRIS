import numpy as np
from typing import Dict
from iris.metrics.base import Metric
from iris.data_types import ModelResponse
from iris.model_wrappers.generative_models.base import GenerativeLLM


class ConsistencyRateMetric(Metric):
    system_prompt = (
        'Determine whether answer "A" is the same or contradicted with the answer "A Reference" for the query "Q".\n'
        'For the tasks with fixed answers, if the two answers are exactly the same you give "Same", otherwise, you give "Contradicted" as the output.\n'
    )
    prompt_template = (
        'Q: {question}\n'
        'A Reference: {reference_answer}\n'
        'A: {answer}\n'
    )

    def __init__(self, judge: GenerativeLLM):
        self.judge = judge
        self.judge.system_prompt = self.system_prompt
        self.judge.post_processing = lambda x: float(1 if x.strip().capitalize() == "Same" else 0)

    def _compute_scores(self, response: ModelResponse) -> Dict:
        scores = []
        for reference_index in range(len(response.answers) - 1):
            for target_index in range(reference_index + 1, len(response.answers)):
                score = self.judge.complete(
                    self.prompt_template.format(
                        question=response.get_prompts()[reference_index],
                        reference_answer=response.answers[reference_index],
                        answer=response.answers[target_index],
                    )
                )
                scores.append(score)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        return {"consistency_rate": {"mean": mean_score, "std": std_score, "all": scores}}


if __name__ == "__main__":
    from llama_index.llms.together import TogetherLLM
    from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


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

    metric = ConsistencyRateMetric(
        judge=APIGenerativeLLM(
            llm=TogetherLLM(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
            ),
            cache_path="./cache",
        )
    )
    result = metric.eval(response)
    print(result)
    assert result.scores["consistency_rate"]["mean"] == 0.42857142857142855
    assert result.scores["consistency_rate"]["std"] == 0.4948716593053935
    print("Passed test!")