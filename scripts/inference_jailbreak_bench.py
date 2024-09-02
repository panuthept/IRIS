import os
import torch
import argparse
from typing import List
from iris.data_types import Sample, ModelResponse
from llama_index.llms.openai_like import OpenAILike
from iris.utilities.loaders import save_model_answers
from iris.datasets.jailbreak_bench import JailbreakBenchDataset
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="harmful")
    args = parser.parse_args()

    dataset = JailbreakBenchDataset(
        attack_engine=args.attack_engine, 
        cache_dir="./data/datasets/jailbreak_bench"
    )
    samples: List[Sample] = dataset.as_samples(split=args.dataset_split)

    if args.api_key:
        model =  APIGenerativeLLM(
            llm=OpenAILike(
                model=args.model_name,
                api_key=args.api_key,
                api_base=args.api_base,
            ),
            cache_path="./cache",
        )
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")
        model = HuggfaceGenerativeLLM(
            args.model_name,
            max_tokens=args.max_tokens,
            pipeline_kwargs={
                "torch_dtype": torch.bfloat16,
                "model_kwargs": {
                    "cache_dir": "./data/models",
                    "local_files_only": False,
                }
            },
            cache_path="./cache",
        )
        print(f"Device: {model.llm.device}")
    responses: List[ModelResponse] = model.complete_batch(samples)

    # Save the responses to a file
    output_dir = f"./outputs/JailbreakBenchDataset/{args.attack_engine}/{args.dataset_split}/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    save_model_answers(responses, f"{output_dir}/response.jsonl")