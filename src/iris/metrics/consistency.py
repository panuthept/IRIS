import numpy as np
from typing import List
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
from iris.data_types import GenerativeLLMResponse, GenerativeLLMResult


class ConsistencyRateMetrics:
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

    def __call__(self, responses: List[GenerativeLLMResponse]):
        results: List[GenerativeLLMResult] = []
        for response in responses:
            result = GenerativeLLMResult.from_response(response)

            avg_score = []
            for answer_variation in response.answer_variations:
                response = self.pipeline.run(
                    question=response.instruction,
                    reference_answer=response.answer,
                    answer=answer_variation,
                ).message.content
                score = float(1 if response == "Same" else 0)
                avg_score.append(score)
            avg_score = np.mean(avg_score)
            result.consistency_score = avg_score
            results.append(result)
        return results


if __name__ == "__main__":
    from iris.data_types import Sample, GenerativeLLMResponse
    from iris.model_wrappers.api_model import APIGenerativeLLM
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.together import TogetherLLM
    from iris.synthesizers.text_synthesizers.paraphrasing_synthesizer import ParaphrasingSynthesizer

    sample = Sample(instruction="Who is the first president of the United States?")
    print(f"Question: {sample.instruction}")

    synthesizer = ParaphrasingSynthesizer(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    sample = synthesizer.synthesize(sample)
    print(f"New question: {sample.instruction_variations}")

    model = APIGenerativeLLM(
        llm=TogetherLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
        )
    )
    responses = model([sample])
    print(f"Response: {responses[0].answer}")
    print(f"Response Variations: {responses[0].answer_variations}")

    metric = ConsistencyRateMetrics(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    results = metric(responses)
    print(results[0])