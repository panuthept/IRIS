import os
import json
import random
import argparse
from tqdm import tqdm
from iris.datasets import load_dataset
from iris.model_wrappers.guard_models import WildGuard

"""
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name WildGuardMixDataset \
--dataset_split train \
--max_samples 4000 \
--save_activations \
--save_logits \
--mask_first_n_tokens 89 \
--mask_last_n_tokens 30 \
--invert_mask \
--output_path ./outputs/wildguard/WildGuardMixDataset/train/4000_prompts_mask_prompt.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention hard_benign \
--save_activations \
--save_logits \
--mask_first_n_tokens 89 \
--mask_last_n_tokens 30 \
--invert_mask \
--output_path ./outputs/wildguard/ORBenchDataset/test/hard_benign_prompts_mask_prompt.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name ORBenchDataset \
--dataset_split test \
--prompt_intention harmful \
--save_activations \
--save_logits \
--mask_first_n_tokens 89 \
--mask_last_n_tokens 30 \
--invert_mask \
--output_path ./outputs/wildguard/ORBenchDataset/test/harmful_prompts_mask_prompt.jsonl
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="WildGuardMixDataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--mask_first_n_tokens", type=int, default=None)
    parser.add_argument("--mask_last_n_tokens", type=int, default=None)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--output_path", type=str, default="./outputs/inference_wildguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # Initial model
    model = WildGuard(
        model_name_or_path=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )

    # Initial dataset
    dataset = load_dataset(args.dataset_name, args.prompt_intention, args.attack_engine)
    samples = dataset.as_samples(split=args.dataset_split)
    random.shuffle(samples)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    # Create save directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    tp = 0
    fp = 0
    fn = 0
    unknown_count = 0
    harmful_count = 0
    benign_count = 0
    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            prompts = sample.get_prompts()
            gold_labels = sample.instructions_true_label
            for prompt, gold_label in zip(prompts, gold_labels):
                pred_labels, pred_tokens = model.generate(
                    prompt, 
                    return_ori_tokens=True,
                    mask_first_n_tokens=args.mask_first_n_tokens,
                    mask_last_n_tokens=args.mask_last_n_tokens,
                    invert_mask=args.invert_mask,
                )

                if gold_label == "Harmful":
                    harmful_count += 1
                    if len(pred_labels) > 0:
                        if pred_labels[0][0] == "Harmful":
                            tp += 1
                        else:
                            fn += 1
                    else:
                        fn += 1
                else:
                    benign_count += 1
                    if len(pred_labels) > 0:
                        if pred_labels[0][0] == "Harmful":
                            fp += 1
                if pred_tokens[0][0] not in [" yes", " no"]:
                    unknown_count += 1

                cache = model.model.logitlens.fetch_cache(return_tokens=args.save_tokens, return_logits=args.save_logits, return_activations=args.save_activations)
                cache = {key: {module_name: activation for module_name, activation in activations.items() if "self_attn" not in module_name and "mlp" not in module_name} for key, activations in cache.items()}
                # Get attention and inputs
                attentions, inputs = model.model.logitlens.fetch_attentions()
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": pred_labels,
                    "label": gold_label,
                    "raw_response": pred_tokens,
                    "cache": cache,
                    "attentions": attentions[0][:, -1].tolist(),    # Only save attention of the last token
                    "inputs": inputs[0].tolist(),
                }, ensure_ascii=False) + "\n")
    # Calculate TPR and FPR
    tpr = tp / (harmful_count + 1e-7)
    fpr = fp / (benign_count + 1e-7)
    fnr = fn / (harmful_count + 1e-7)
    print(f"TPR: {round(tpr, 2)}")
    print(f"FPR: {round(fpr, 2)}")
    print(f"FNR: {round(fnr, 2)}")
    print(f"Unknown count: {unknown_count}")