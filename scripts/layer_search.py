import os
import json
import random
import argparse
from tqdm import tqdm
from iris.datasets import WildGuardMixDataset
from iris.model_wrappers.guard_models import WildGuard


"""
CUDA_VISIBLE_DEVICES=0 python scripts/layer_search.py \
--model_name allenai/wildguard \
--save_path ./layer_search_outputs/wildguard/responses.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/layer_search.py \
--model_name allenai/wildguard \
--checkpoint_path ./finetuned_models/iris_wildguard_layer_19/checkpoint-1220 \
--save_path ./layer_search_outputs/iris_wildguard_layer_19/responses.jsonl
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./layer_search_outputs/wildguard/responses.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load model
    model = WildGuard(
        model_name_or_path=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )
    
    # Load dataset
    dataset = WildGuardMixDataset(attack_engine="vanilla")
    samples = dataset.as_samples(split="train")

    # Get development set
    random.shuffle(samples)
    if args.train_eval_split == 1.0:
        train_samples, eval_samples = samples, []
    else:
        train_size = int(len(samples) * args.train_eval_split)
        train_samples, eval_samples = samples[:train_size], samples[train_size:]
    print(f"Dev size: {len(eval_samples)}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as f:
            for sample in tqdm(samples):
                prompts = sample.get_prompts()
                gold_labels = sample.instructions_true_label
                for prompt, gold_label in zip(prompts, gold_labels):
                    response = model.generate(prompt, return_probs=True)
                    cache = model.model.logitlens.fetch_cache(
                        return_tokens=False, 
                        return_logits=True, 
                        return_activations=True,
                    )
                    f.write(json.dumps({
                        "prompt": prompt,
                        "response": response,
                        "label": gold_label,
                        "cache": cache
                    }, ensure_ascii=False) + "\n")