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
CUDA_VISIBLE_DEVICES=0 python scripts/inference_safeguard.py \
--safeguard_name ShieldGemma \
--model_name google/shieldgemma-9b \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language vi \
--disable_logitlens \
--output_path ./outputs/ShieldGemma9B/SEASafeguardDataset/vi/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_safeguard.py \
--safeguard_name LlamaGuard \
--model_name meta-llama/Llama-Guard-3-8B \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language vi \
--disable_logitlens \
--output_path ./outputs/LlamaGuard8B/SEASafeguardDataset/vi/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language en \
--disable_logitlens \
--output_path ./outputs/WildGuard/SEASafeguardDataset/en/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=0,1 python scripts/inference_safeguard.py \
--safeguard_name WildGuard \
--model_name allenai/wildguard \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language my \
--disable_logitlens \
--output_path ./outputs/WildGuard/SEASafeguardDataset/my/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name SEASafeguardDataset \
--dataset_split test \
--language vi \
--disable_logitlens \
--output_path ./outputs/NemoGuard/SEASafeguardDataset/vi/test/all_prompts.jsonl




CUDA_VISIBLE_DEVICES=1 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name WildGuardMixDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/WildGuardMixDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name ORBenchDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/ORBenchDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name OpenAIModerationDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/OpenAIModerationDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name ToxicChatDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/ToxicChatDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=2 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name XSTestDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/XSTestDataset/test/all_prompts.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/inference_safeguard.py \
--safeguard_name NemoGuard \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--checkpoint_path ./data/models/llama-3.1-nemoguard-8b-content-safety-lora-adapter \
--dataset_name JailbreakBenchDataset \
--dataset_split test \
--top_logprobs 2 \
--save_logits \
--save_activations \
--disable_logitlens \
--output_path ./outputs/NemoGuard/JailbreakBenchDataset/test/all_prompts.jsonl
"""

def prompt_intervention(prompt, preserve_tokens, tokenizer):
    intervented_prompt_tokens = []
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for token_id in prompt_tokens:
        if token_id not in preserve_tokens:
            continue
        intervented_prompt_tokens.append(token_id)
    return tokenizer.decode(intervented_prompt_tokens)

def prompt_intervention_2(prompt):
    suffled_prompt = prompt.split()
    random.shuffle(suffled_prompt)
    return " ".join(suffled_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safeguard_name", type=str, default="LlamaGuard", choices=list(AVAILABLE_GUARDS.keys()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="SEASafeguardDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_logprobs", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_intention", type=str, default=None)
    parser.add_argument("--attack_engine", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--save_attentions", action="store_true")
    parser.add_argument("--disable_logitlens", action="store_true")
    parser.add_argument("--mask_first_n_tokens", type=int, default=None)
    parser.add_argument("--mask_last_n_tokens", type=int, default=None)
    parser.add_argument("--mask_topk_tokens", type=int, default=None)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--prompt_intervention", action="store_true")
    parser.add_argument("--prompt_intervention_2", action="store_true")
    parser.add_argument("--lmi_label", type=str, default="Harmful", choices=["Harmful", "Safe"])
    parser.add_argument("--lmi_threshold", type=float, default=0.0)
    parser.add_argument("--lmi_k", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="./outputs/inference_safeguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    with open("./data/LMI_scores.jsonl", "r") as f:
        data = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
        train_harmful_LMIs = [(k, v) for k, v in data["LMI"]["harmful"].items() if v > args.lmi_threshold][:args.lmi_k]
        train_harmful_LMIs = {int(k): v for k, v in train_harmful_LMIs}
        train_safe_LMIs = [(k, v) for k, v in data["LMI"]["benign"].items() if v > args.lmi_threshold][:args.lmi_k]
        train_safe_LMIs = {int(k): v for k, v in train_safe_LMIs}

    # Get mask_tokens
    mask_tokens = None
    if args.mask_topk_tokens is not None:
        with open("./data/LMI_scores.jsonl", "r") as f:
            data = json.load(f)
            train_harmful_LMIs = list(data["LMI"]["harmful"])[:args.mask_topk_tokens]
            mask_tokens = [int(k) for k in train_harmful_LMIs]
        print(f"Mask tokens: {mask_tokens}")

    # Initial safeguard
    safeguard = load_safeguard(
        safeguard_name=args.safeguard_name,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        disable_logitlens=args.disable_logitlens,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )
    # print(len(safeguard.prefix_token_ids))
    # print(len(safeguard.suffix_token_ids))

    # Initial dataset
    dataset = load_dataset(args.dataset_name, args.prompt_intention, args.attack_engine, args.language)
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
                # Intervene prompt (Optional)
                if args.prompt_intervention:
                    if args.lmi_label == "Harmful":
                        prompt = prompt_intervention(prompt, train_harmful_LMIs, tokenizer)
                    else:
                        prompt = prompt_intervention(prompt, train_safe_LMIs, tokenizer)
                elif args.prompt_intervention_2:
                    prompt = prompt_intervention_2(prompt)
                pred_labels, pred_tokens = safeguard.generate(
                    prompt, 
                    return_ori_tokens=True,
                    mask_first_n_tokens=args.mask_first_n_tokens,
                    mask_last_n_tokens=args.mask_last_n_tokens,
                    mask_tokens=mask_tokens,
                    invert_mask=args.invert_mask,
                )
                # if pred_tokens[0][0] not in safeguard.valid_outputs:
                #     unknown_count += 1

                gold_label = int(gold_label == "Harmful")
                # NOTE: This implementation handles the case when the safeguard predicts invalid token
                pred_score = 0.0
                if len(pred_labels) == 2:
                    if len(pred_labels[0]) == 2:
                        _pred_scores = [score for label, score in pred_labels if label == "Harmful"]
                        if len(_pred_scores) > 0:
                            pred_score = _pred_scores[0]
                    else:
                        _pred_scores = [score for label, score, logit in pred_labels if label == "Harmful"]
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
                    "attentions": attentions[0].tolist() if not args.disable_logitlens and args.save_attentions else None, # Only save the last token attentions
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