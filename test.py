from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from iris.metrics.consistency import ConsistencyRateMetric
from iris.data_types import Sample, ModelResponse, EvaluationResult
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


if __name__ == "__main__":
    # Dataset: List[Sample] -> Model: List[ModelResponse] -> Metric: List[EvaluationResult]
    samples: List[Sample] = [
        Sample(
            instructions=[
                "Output whether the sentiment of the input sentence is positive or negative.",
                "Given an input text, output whether the sentiment is positive or negative.",
                "For each input, determine if the sentiment in the input is prone to negative or positive opinion.",
                "For each input, determine whether it expresses a positive or a negative opinion.",
                "Classify the sentiment of the input sentence (options are positive or negative)",
                "write \"positive\" if the input is a positive review, and \"negative\" if the input is a negative review",
                "Determine whether the sentiment is positive or negative",
                "Output whether the sentiment is positive or negative"
            ],
            query="A tender, heartfelt family drama.",
            reference_answers=["Positive"],
        )
    ]

    model = APIGenerativeLLM(
        llm=TogetherLLM(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            model="Qwen/Qwen1.5-0.5B-Chat",
            api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
        ),
        system_prompt="You will be given a instruction and query, use deterministic output as 'Positive' or 'Negative' without additional information or character.",
        post_processing=lambda x: x.strip().capitalize(),
    )
    responses: List[ModelResponse] = model.complete_batch(samples)
    print(f"Responses: {responses[0].answers}")

    metric = ConsistencyRateMetric(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    results: List[EvaluationResult] = metric.eval_batch(responses, reference_index = 0)
    print(results)