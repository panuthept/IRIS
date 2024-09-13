import os
import torch
import argparse
from iris.model_wrappers.guard_models import LlamaGuard, WildGuard
from iris.augmentations.instruction_augmentations.jailbreaks import MultiLingualJailbreaking
from iris.datasets import (
    JailbreakBenchDataset,
    JailbreaKV28kDataset,
    WildGuardMixDataset,
    XSTestDataset,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--dataset_name", type=str, default="jailbreak_bench")
    parser.add_argument("--jailbreak_name", type=str, default="multilingual")
    parser.add_argument("--intention", type=str, default="harmful")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    args = parser.parse_args()

    assert args.dataset_name in ["jailbreak_bench", "jailbreakv_28k", "wildguardmix", "xstest"]
    assert args.jailbreak_name in ["multilingual", "multilingual+"]

    if args.dataset_name == "jailbreak_bench":
        dataset = JailbreakBenchDataset(intention=args.intention)
    elif args.dataset_name == "jailbreakv_28k":
        dataset = JailbreaKV28kDataset(intention=args.intention)
    elif args.dataset_name == "wildguardmix":
        dataset = WildGuardMixDataset(intention=args.intention)
    else:
        dataset = XSTestDataset(intention=args.intention)
    samples = dataset.as_samples()
    print(f"Running dataset: {args.dataset_name}")

    # Get model
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.model_name.startswith("meta-llama"):
        target_model = LlamaGuard(
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
            use_cache=args.use_cache,
            cache_path="./cache",
        )
    elif args.model_name == "allenai/wildguard":
        target_model = WildGuard(
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
            use_cache=args.use_cache,
            cache_path="./cache",
        )
    else:
        raise ValueError(f"Model {args.model_name} not supported.")
    print(f"Device: {target_model.device}")

    # Get jailbreak
    if args.jailbreak_name == "multilingual":
        augmentation = MultiLingualJailbreaking(
            target_model=target_model, 
            evaluator=lambda x: x.strip().capitalize() == "Benign",
            apply_jailbreak_template=False,
            translate_answer_to_en=False,
            use_cache=args.use_cache,
            cache_path="./cache",
        )
    elif args.jailbreak_name == "multilingual+":
        augmentation = MultiLingualJailbreaking(
            target_model=target_model, 
            evaluator=lambda x: x.strip().capitalize() == "Benign",
            apply_jailbreak_template=True,
            translate_answer_to_en=False,
            use_cache=args.use_cache,
            cache_path="./cache",
        )
    else:
        raise ValueError(f"Jailbreak {args.jailbreak_name} not supported.")
    
    augmentation.augment_batch(samples)
    print(f"ASR: {augmentation.attack_success_rate}")
    print(f"Supports: {len(samples)}")