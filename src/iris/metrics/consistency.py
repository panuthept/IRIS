import numpy as np
from typing import Dict
from iris.metrics.base import Metric
from iris.data_types import ModelResponse
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline


class ConsistencyRateMetric(Metric):
    prompt_template = (
        'Determine whether answer "A" is the same or contradicted with the answer "A Reference" for the query "Q".\n'
        'For the tasks with fixed answers, if the two answers are exactly the same you give "Same", otherwise, you give "Contradicted" as the output.\n\n'
        'Q: {question}\n'
        'A Reference: {reference_answer}\n'
        'A: {answer}\n'
    )

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = PromptTemplate(self.prompt_template)
        self.pipeline = QueryPipeline(chain=[self.system_prompt, self.llm])

    def _compute_scores(self, response: ModelResponse, reference_index: int = 0) -> Dict:
        avg_score = []
        for idx, answer in enumerate(response.answers):
            if idx == reference_index:
                continue
            result = self.pipeline.run(
                question=response.get_prompts()[reference_index],
                reference_answer=response.answers[reference_index],
                answer=answer,
            ).message.content
            score = float(1 if result == "Same" else 0)
            avg_score.append(score)
        avg_score = np.mean(avg_score)
        return {"consistency_rate": avg_score}


if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI


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
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    result = metric.eval(response)
    print(result)
    assert result.scores["consistency_rate"] == 3/7
    print("Passed test!")