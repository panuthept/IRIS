import os
import torch
import argparse
from iris.benchmarks import JailbreakBenchBenchmark
from llama_index.llms.openai_like import OpenAILike
from iris.model_wrappers.generative_models.api_model import APIGenerativeLLM
from iris.model_wrappers.generative_models.huggingface_model import HuggfaceGenerativeLLM
from iris.model_wrappers.generative_models.transformer_lens_model import TransformerLensGenerativeLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--intervention", action="store_true")
    parser.add_argument('--intervention_layers', nargs='+', type=int, default=[19, 20, 21, 22])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    benchmark = JailbreakBenchBenchmark()

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
            if args.intervention:
                model = TransformerLensGenerativeLLM(
                    args.model_name,
                    max_tokens=args.max_tokens,
                    activation_layers=args.intervention_layers,
                    from_pretrained_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "cache_dir": "./data/models",
                        "local_files_only": False,
                    },
                    cache_path="./cache",
                    use_cache=False,
                )
                # TODO: Print the device of the model
            else:
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
        print(f"{task}:")
        print(f"Refusal Rate: {round(task_results['refusal_rate']['mean_all'], 2)}")
        print(f"Safe Response Rate: {round(task_results['safe_response_rate']['mean_all'], 2)}")
        print("-" * 100)