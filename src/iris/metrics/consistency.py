import numpy as np
from typing import List
from llama_index.core import PromptTemplate
from iris.data_types import Response, Result
from llama_index.core.query_pipeline import QueryPipeline


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

    def __call__(self, reference_responses: List[Response], target_responses: List[Response]):
        assert len(reference_responses) == len(target_responses), "The number of reference and target responses should be the same. Got {} and {}.".format(len(reference_responses), len(target_responses))

        results: List[Result] = []
        for reference_response, target_response in zip(reference_responses, target_responses):
            avg_score = []
            for reference_answer in reference_response.responses:
                for target_answer in target_response.responses:
                    response = self.pipeline.run(
                        question=reference_response.instruction,
                        reference_answer=reference_answer,
                        answer=target_answer,
                    ).message.content
                    score = float(1 if response == "Same" else 0)
                    avg_score.append(score)
            avg_score = np.mean(avg_score)
            result = Result.from_response(target_response, score=avg_score)
            results.append(result)
        return results


if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.together import TogetherLLM
    from iris.synthesizers.text_synthesizers.paraphrasing_synthesizer import ParaphrasingSynthesizer

    question = "Who is the first president of the United States?"
    print(f"Question: {question}")

    synthesizer = ParaphrasingSynthesizer(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    question_variants = synthesizer.synthesize(question)
    print(f"New question: {question_variants}")

    llm = TogetherLLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
    )
    ref_resp = llm.complete(question).text
    print(f"Reference response: {ref_resp}")
    tar_resp = llm.complete(question_variants[0]).text
    print(f"Target response: {tar_resp}")

    ref_resp = [Response(instruction=question, responses=[ref_resp])]
    tar_resp = [Response(instruction=question_variants[0], responses=[tar_resp])]

    metric = ConsistencyRateMetrics(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    result = metric(ref_resp, tar_resp)
    print(result)