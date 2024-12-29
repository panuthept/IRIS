import os
import json
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from iris.datasets import load_dataset, AVAILABLE_DATASETS
from iris.metrics.safeguard_metrics import SafeGuardMetric
from iris.model_wrappers.guard_models import load_safeguard, AVAILABLE_GUARDS

"""
CUDA_VISIBLE_DEVICES=0 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name WildGuardMixDataset \
--dataset_split test \
--top_logprobs 128 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/WildGuardMixDataset/test/all_prompts_bias_model.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name ORBenchDataset \
--dataset_split test \
--top_logprobs 128 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/ORBenchDataset/test/all_prompts_bias_model.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name OpenAIModerationDataset \
--dataset_split test \
--top_logprobs 128 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/OpenAIModerationDataset/test/all_prompts_bias_model.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name ToxicChatDataset \
--dataset_split test \
--top_logprobs 128 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/ToxicChatDataset/test/all_prompts_bias_model.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name XSTestDataset \
--dataset_split test \
--top_logprobs 1024 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/XSTestDataset/test/all_prompts_bias_model.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_bias_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name JailbreakBenchDataset \
--dataset_split test \
--top_logprobs 1024 \
--threshold 0.0 \
--k 1000000 \
--output_path ./outputs/WildGuard/JailbreakBenchDataset/test/all_prompts_bias_model.jsonl
"""


def prompt_intervention(prompt, remove_tokens, tokenizer):
    intervented_prompt_tokens = []
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for token_id in prompt_tokens:
        if token_id not in remove_tokens:
            continue
        intervented_prompt_tokens.append(token_id)
    return tokenizer.decode(intervented_prompt_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safeguard_name", type=str, default="WildGuard", choices=list(AVAILABLE_GUARDS.keys()))
    parser.add_argument("--model_name", type=str, default="allenai/wildguard")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="WildGuardMixDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=100000)
    parser.add_argument("--top_logprobs", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--disable_logitlens", action="store_true")
    parser.add_argument("--mask_first_n_tokens", type=int, default=None)
    parser.add_argument("--mask_last_n_tokens", type=int, default=None)
    parser.add_argument("--mask_topk_tokens", type=int, default=None)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--output_path", type=str, default="./outputs/inference_bias_safeguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    with open("./data/LMI_scores.jsonl", "r") as f:
        data = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
        train_harmful_LMIs = [(k, v) for k, v in data["LMI"]["harmful"].items() if v > args.threshold][:args.k]
        train_harmful_LMIs = {int(k): v for k, v in train_harmful_LMIs}

    # Initial safeguard
    safeguard = load_safeguard(
        safeguard_name=args.safeguard_name,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        disable_logitlens=args.disable_logitlens,
        top_logprobs=args.top_logprobs,
    )
    print(len(safeguard.prefix_token_ids))
    print(len(safeguard.suffix_token_ids))

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
                prompt = prompt_intervention(prompt, train_harmful_LMIs, tokenizer)
                pred_labels, pred_tokens = safeguard.generate(
                    prompt, 
                    return_ori_tokens=True,
                    mask_first_n_tokens=args.mask_first_n_tokens,
                    mask_last_n_tokens=args.mask_last_n_tokens,
                    invert_mask=args.invert_mask,
                )

                gold_label = int(gold_label == "Harmful")
                # NOTE: This implementation handles the case when the safeguard predicts invalid token
                pred_score = 0.0
                if len(pred_labels) == 2:
                    if len(pred_labels[0]) == 2:
                        _pred_scores = [score for label, score in pred_labels if label == "Harmful"]
                        if len(_pred_scores) > 0:
                            pred_score = _pred_scores[0]
                    else:
                        _pred_scores = [score for label, score, _ in pred_labels if label == "Harmful"]
                        if len(_pred_scores) > 0:
                            pred_score = _pred_scores[0]
                else:
                    unknown_count += 1

                gold_labels.append(gold_label)
                pred_scores.append(pred_score)

                if not args.disable_logitlens:
                    cache = safeguard.model.logitlens.fetch_cache(return_tokens=args.save_tokens, return_logits=args.save_logits, return_activations=args.save_activations)
                    cache = {key: {module_name: activation for module_name, activation in activations.items() if "self_attn" not in module_name and "mlp" not in module_name} for key, activations in cache.items()}
                    # Get attention and inputs
                    attentions, inputs = safeguard.model.logitlens.fetch_attentions()
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": pred_labels,
                    "label": "Harmful" if gold_label == 1 else "Benign",
                    "raw_response": pred_tokens,
                    "cache": cache if not args.disable_logitlens else None,
                    "attentions": attentions[0][:, -1].tolist() if not args.disable_logitlens else None,    # Only save attention of the last token
                    "inputs": inputs[0].tolist() if not args.disable_logitlens else None,
                }, ensure_ascii=False) + "\n")
    # Calculate metrics
    metrics = SafeGuardMetric()
    metrics.update(gold_labels, pred_scores)
    print(f"Recall: {round(metrics.recall * 100, 1)}")
    print(f"Precision: {round(metrics.precision * 100, 1)}")
    print(f"F1: {round(metrics.f1 * 100, 1)}")
    print(f"AUPRC: {round(metrics.pr_auc * 100, 1)}")
    print(f"Unknown count: {unknown_count}")