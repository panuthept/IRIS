import json
import torch
import random
import argparse
from tqdm import tqdm
from trl import SFTConfig
from iris.datasets import JailbreakBenchDataset
from iris.model_wrappers.guard_models import WildGuard


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    # parser.add_argument("--attack_engine", type=str, default=None)
    # parser.add_argument("--cache_dir", type=str, default="./data/datasets/wildguardmix")
    # parser.add_argument("--train_eval_split", type=float, default=0.9)
    # parser.add_argument("--max_seq_length", type=int, default=8192)
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--learning_rate", type=float, default=2e-6)
    # parser.add_argument("--weight_decay", type=float, default=0.00)
    # parser.add_argument("--warmup_ratio", type=float, default=0.03)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--epochs", type=int, default=2)
    # parser.add_argument("--eval_steps", type=int, default=10)
    # parser.add_argument("--output_dir", type=str, default="./finetuned_models/sft_wildguard")
    # parser.add_argument("--report_to", type=str, default="all")
    # parser.add_argument("--allow_cpu", action="store_true")
    # args = parser.parse_args()

    # random.seed(args.seed)

    model = WildGuard(model_name_or_path="allenai/wildguard")

    # Harmful prompts
    dataset = JailbreakBenchDataset(intention="harmful")
    samples = dataset.as_samples(split="test")

    for sample in tqdm(samples):
        prompts = sample.get_prompts()
        labels = ["Harmful"] * len(prompts)
        for prompt, label in zip(prompts, labels):
            response = model.generate(prompt, return_probs=False)
            activations = model.model.logitlens.get_last_activations()
            with open("./harmful_prompts.jsonl", "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "activations": activations
                }, ensure_ascii=False) + "\n")

    # Benign prompts
    dataset = JailbreakBenchDataset(intention="benign")
    samples = dataset.as_samples(split="test")

    for sample in tqdm(samples):
        prompts = sample.get_prompts()
        labels = ["Benign"] * len(prompts)
        for prompt, label in zip(prompts, labels):
            response = model.generate(prompt, return_probs=False)
            activations = model.model.logitlens.get_last_activations()
            with open("./benign_prompts.jsonl", "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "activations": activations
                }, ensure_ascii=False) + "\n")