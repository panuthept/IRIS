import os
import torch
import argparse
from iris.models.llama_guard import LlamaGuard
from iris.benchmarks import JailbreakBenchPromptCLFBenchmark

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--eval_only", action="store_true")
    # parser.add_argument("--intervention", action="store_true")
    # parser.add_argument('--intervention_layers', nargs='+', type=int, default=[19, 20, 21, 22])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    args = parser.parse_args()

    benchmark = JailbreakBenchPromptCLFBenchmark()

    # Get model
    model = None
    if not args.eval_only:
        print(f"CUDA available: {torch.cuda.is_available()}")
        model = LlamaGuard(
            huggingface_model_name_or_path=args.model_name,
            pipeline_kwargs={
                "torch_dtype": torch.bfloat16,
                "model_kwargs": {
                    "cache_dir": "./data/models",
                    "local_files_only": False,
                }
            },
            cache_path="./cache",
        )
        print(f"Device: {model.device}")

    benchmark_results = benchmark.evaluate(model=model, model_name=args.model_name)
    for task, task_results in benchmark_results.items():
        print(f"{task}:")
        print(f"Refusal Rate: {round(task_results['refusal_rate']['mean_all'], 2)}")
        print(f"Safe Response Rate: {round(task_results['safe_response_rate']['mean_all'], 2)}")
        print("-" * 100)