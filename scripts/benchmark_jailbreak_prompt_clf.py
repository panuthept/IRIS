import os
import torch
import argparse
from iris.model_wrappers.guard_models import LlamaGuard, WildGuard
from iris.benchmarks import (
    JailbreakBenchPromptCLFBenchmark, 
    JailbreaKV28kPromptCLFBenchmark, 
    WildGuardMixPromptCLFBenchmark,
    XSTestPromptCLFBenchmark,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--benchmark_name", type=str, default="jailbreak_bench")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    args = parser.parse_args()

    assert args.benchmark_name in ["jailbreak_bench", "jailbreakv_28k", "wildguardmix", "xstest"]

    if args.benchmark_name == "jailbreak_bench":
        benchmark = JailbreakBenchPromptCLFBenchmark()
    elif args.benchmark_name == "jailbreakv_28k":
        benchmark = JailbreaKV28kPromptCLFBenchmark()
    elif args.benchmark_name == "wildguardmix":
        benchmark = WildGuardMixPromptCLFBenchmark()
    else:
        benchmark = XSTestPromptCLFBenchmark()
    print(f"Running benchmark: {args.benchmark_name}")

    # Get model
    model = None
    if not args.eval_only:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if args.model_name.startswith("meta-llama"):
            model = LlamaGuard(
                huggingface_model_name_or_path=args.model_name,
                api_key=args.api_key,
                api_base=args.api_base,
                pipeline_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "model_kwargs": {
                        "cache_dir": "./data/models",
                        "local_files_only": False,
                    }
                },
                cache_path="./cache",
                post_processing=lambda x: "Benign" if x.strip().capitalize() == "Safe" else "Harmful",
            )
        elif args.model_name == "allenai/wildguard":
            model = WildGuard(
                huggingface_model_name_or_path=args.model_name,
                api_key=args.api_key,
                api_base=args.api_base,
                pipeline_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "model_kwargs": {
                        "cache_dir": "./data/models",
                        "local_files_only": False,
                    }
                },
                cache_path="./cache",
            )
        else:
            raise ValueError(f"Model {args.model_name} not supported.")
        print(f"Device: {model.device}")

    benchmark_results = benchmark.evaluate(model=model, model_name=args.model_name)
    if args.inference_only:
        print("Done inference only mode.")
    else:
        for task, task_results in benchmark_results.items():
            print(f"{task}:")
            print(f"Accuracy: {round(task_results['exact_match']['mean_all'], 2)}")
            print(f"Supports: {task_results['exact_match']['supports_all']}")
            print("-" * 100)