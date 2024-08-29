from iris.data_types import Sample
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from iris.metrics.consistency import ConsistencyRateMetric
from iris.utilities.loaders import save_model_answers, load_model_answers
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.synthesizers.text_synthesizers.paraphrasing_synthesizer import ParaphrasingSynthesizer


if __name__ == "__main__":
    from iris.data_types import Sample
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.together import TogetherLLM
    from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
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
    response = model.complete(sample)
    print(f"Response: {response.answer}")
    print(f"Response Variations: {response.answer_variations}")

    metric = ConsistencyRateMetric(
        llm=OpenAI(
            model="gpt-4o", 
            api_key="sk-proj-uvbi9yfICRLlEdB9WuVLT3BlbkFJLI51rD9gebE9T5pxxztV",
        ),
    )
    result = metric.eval(response)
    print(result)

    # benchmark = AlpacaEvalBenchmark()
    # test_samples: List[Sample] = AlpacaEvalBenchmark.get_test_set()

    # model = HuggfaceInferenceLLM(
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     system_prompt=None,
    # )
    # responses: List[GenerativeLLMResponse] = model(test_samples)
    # save_model_answers(responses, "./outputs/model_answers.jsonl")

    # model_answers: List[GenerativeLLMResponse] = load_model_answers("./outputs/model_answers.jsonl")
    # results: List[GenerativeLLMResult] = benchmark.evaluate(model_answers)