import numpy as np
from typing import Dict
from iris.metrics.base import Metric
from iris.data_types import ModelResponse
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline


class ConsistencyRateMetric(Metric):
    prompt_template = (
        'Determine whether answer "A" is the same or contradicted with the answer "A Reference" for the question "Q".\n'
        'For the tasks with fixed answers, if the two answers are exactly the same you give "Same", otherwise, you give "Contradicted" as the output.\n\n'
        'Q: {question}\n'
        'A Reference: {reference_answer}\n'
        'A: {answer}\n'
    )

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = PromptTemplate(self.prompt_template)
        self.pipeline = QueryPipeline(chain=[self.system_prompt, self.llm])

    def _compute_scores(self, response: ModelResponse) -> Dict:
        avg_score = []
        for answer_variation in response.answer_variations:
            result = self.pipeline.run(
                question=response.instruction,
                reference_answer=response.answer,
                answer=answer_variation,
            ).message.content
            score = float(1 if result == "Same" else 0)
            avg_score.append(score)
        avg_score = np.mean(avg_score)
        return {"consistency_rate": avg_score}


if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI


    response = ModelResponse(
        instruction="Who is the first president of the United States?",
        answer="The first president of the United States was George Washington. He was inaugurated on April 30, 1789, and served two terms in office until March 4, 1797.",
        answer_variations=[
            "The first president of the United States was Barack Obama. He was inaugurated on April 30, 1789, and served two terms in office until March 4, 1797.",
            "The inaugural president of the United States was George Washington. He was inaugurated on April 30, 1789, at Federal Hall in New York City, which was the temporary capital of the United States at the time.",
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
    assert result.scores["consistency_rate"] == 0.5
    print("Passed test!")