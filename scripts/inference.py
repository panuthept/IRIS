import os
import torch
from typing import List
from iris.data_types import Sample, ModelResponse
from iris.utilities.loaders import save_model_answers
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = JailbreakBenchDataset(attack_engine="PAIR")
    samples: List[Sample] = dataset.as_samples(split="harmful")

    model = HuggfaceGenerativeLLM(
        "Qwen/Qwen2-0.5B-Instruct",
        max_tokens=512,
        pipeline_kwargs={
            "torch_dtype": torch.bfloat16,
            "model_kwargs": {
                "cache_dir": "./models",
                "local_files_only": False,
            }
        },
        cache_path="./cache",
    )
    responses: List[ModelResponse] = model.complete_batch(samples)

    # Save the responses to a file
    os.makedirs("./outputs/Qwen/Qwen2-0.5B-Instruct", exist_ok=True)
    save_model_answers(responses, "./outputs/Qwen/Qwen2-0.5B-Instruct/response.jsonl")