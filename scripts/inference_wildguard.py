import os
import json
import argparse
from tqdm import tqdm
from iris.datasets import load_dataset
from iris.model_wrappers.guard_models import WildGuard

"""
CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name WildGuardMixDataset \
--prompt_intention harmful \
--attack_engine vanilla \
--dataset_split train \
--save_tokens \
--output_path ./outputs/wildguard/WildGuardMixDataset/train/vanilla_harmful_prompts.jsonl
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="WildGuardMixDataset")
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--output_path", type=str, default="./outputs/inference_wildguard.jsonl")
    args = parser.parse_args()

    # Initial model
    model = WildGuard(
        model_name_or_path=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )

    # Initial dataset
    dataset = load_dataset(args.dataset_name, args.prompt_intention, args.attack_engine)
    samples = dataset.as_samples(split=args.dataset_split)

    # Create save directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            prompts = sample.get_prompts()
            labels = sample.instructions_true_label
            for prompt, label in zip(prompts, labels):
                response = model.generate(prompt, return_probs=True)
                cache = model.model.logitlens.fetch_cache(return_tokens=args.save_tokens, return_logits=args.save_logits, return_activations=args.save_activations)
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                    "cache": cache
                }, ensure_ascii=False) + "\n")