from typing import List
from llama_index.llms.openai import OpenAI
from iris.data_types import Sample, ModelResponse
from llama_index.llms.together import TogetherLLM
from iris.metrics.jailbreaking import RefusalRateMetric
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM


if __name__ == "__main__":
    # Dataset: List[Sample] -> Model: List[ModelResponse] -> Metric: Tuple[List[EvaluationResult], SummarizedResult]
    dataset = JailbreakBenchDataset(attack_engine="PAIR")
    samples: List[Sample] = dataset.as_samples(split="harmful")[:1]
    print(samples[0].get_prompts()[0])
    print(samples[0].reference_answers)

    model = APIGenerativeLLM(
        llm=TogetherLLM(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            model="meta-llama/Llama-2-13b-chat-hf",
            api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
        ),
    )
    responses: List[ModelResponse] = model.complete_batch(samples)
    print(f"Responses: {responses[0].answers}")

    metric = RefusalRateMetric(
        llm=TogetherLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key="efaa563e1bb5b11eebdf39b8327337113b0e8b087c6df22e2ce0b2130e4aa13f",
        ),
    )
    all_results, summarized_result = metric.eval_batch(responses)
    print(summarized_result.scores)
    print(summarized_result.supports)