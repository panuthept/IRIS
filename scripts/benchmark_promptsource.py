import os
import torch
import argparse
from llama_index.llms.openai_like import OpenAILike
from iris.benchmarks import PromptSourceBenchmark
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    benchmark = PromptSourceBenchmark()

    # Get model
    model = None
    if not args.eval_only:
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

    benchmark_results = benchmark.evaluate(model=model, model_name=args.model_name)
    for task, task_results in benchmark_results.items():
        print(f"Task: {task}")
        print(task_results)
        print("-" * 100)