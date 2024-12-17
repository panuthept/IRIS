import os
import json
import random
import argparse
from tqdm import tqdm
from iris.datasets import load_dataset
from iris.model_wrappers.guard_models import WildGuard
from iris.metrics.safeguard_metrics import SafeGuardMetric

"""
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name WildGuardMixDataset \
--dataset_split test \
--output_path ./outputs/wildguard/WildGuardMixDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name ORBenchDataset \
--dataset_split test \
--output_path ./outputs/wildguard/ORBenchDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name OpenAIModerationDataset \
--dataset_split test \
--output_path ./outputs/wildguard/OpenAIModerationDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name WildChatDataset \
--dataset_split test \
--output_path ./outputs/wildguard/WildChatDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name ToxicChatDataset \
--dataset_split test \
--output_path ./outputs/wildguard/ToxicChatDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name XSTestDataset \
--dataset_split test \
--output_path ./outputs/wildguard/XSTestDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_wildguard.py \
--model_name allenai/wildguard \
--dataset_name JailbreakBenchDataset \
--dataset_split test \
--output_path ./outputs/wildguard/JailbreakBenchDataset/test/all_prompts.jsonl
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
    parser.add_argument("--save_activations", action="store_true")
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

    gold_labels = []
    pred_scores = []
    unknown_count = 0
    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            prompts = sample.get_prompts()
            _gold_labels = sample.instructions_true_label
            for prompt, gold_label in zip(prompts, _gold_labels):
                pred_labels, pred_tokens = model.generate(
                    prompt, 
                    return_ori_tokens=True,
                    mask_first_n_tokens=args.mask_first_n_tokens,
                    mask_last_n_tokens=args.mask_last_n_tokens,
                    invert_mask=args.invert_mask,
                )
                if pred_tokens[0][0] not in [" yes", " no"]:
                    unknown_count += 1

                gold_label = int(gold_label == "Harmful")
                # NOTE: This implementation handles the case when the model predicts invalid token
                pred_score = 0.0
                if len(pred_labels) > 0:
                    if len(pred_labels[0]) == 2:
                        _pred_scores = [score for label, score in pred_labels if label == "Harmful"]
                        if len(_pred_scores) > 0:
                            pred_score = _pred_scores[0]
                    else:
                        _pred_scores = [score for label, score, _ in pred_labels if label == "Harmful"]
                        if len(_pred_scores) > 0:
                            pred_score = _pred_scores[0]

                gold_labels.append(gold_label)
                pred_scores.append(pred_score)

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
    # Calculate metrics
    metrics = SafeGuardMetric()
    metrics.update(gold_labels, pred_scores)
    print(f"Recall: {round(metrics.recall * 100, 2)}")
    print(f"Precision: {round(metrics.precision * 100, 2)}")
    print(f"F1: {round(metrics.f1 * 100, 2)}")
    print(f"AUPRC: {round(metrics.pr_auc * 100, 2)}")
    print(f"Unknown count: {unknown_count}")