from typing import List
from iris.data_types import Sample, Response, Result
from iris.benchmarks.alpaca_eval import AlpacaEvalBenchmark
from iris.model_wrappers.huggingface_model import HuggfaceInferenceLLM
from iris.utilities.loaders import save_model_answers, load_model_answers


if __name__ == "__main__":
    benchmark = AlpacaEvalBenchmark()
    test_samples: List[Sample] = AlpacaEvalBenchmark.get_test_set()

    model = HuggfaceInferenceLLM(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        system_prompt=None,
    )
    responses: List[Response] = model(test_samples)
    save_model_answers(responses, "./outputs/model_answers.jsonl")

    model_answers: List[Response] = load_model_answers("./outputs/model_answers.jsonl")
    results: List[Result] = benchmark.evaluate(model_answers)