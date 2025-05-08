import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from typing import List
# from transformers import AutoTokenizer
from iris.datasets import load_dataset, AVAILABLE_DATASETS
from iris.metrics.safeguard_metrics import SafeGuardMetric
from iris.data_types import SafeGuardInput, SafeGuardResponse
from iris.model_wrappers.guard_models import load_safeguard, AVAILABLE_GUARDS


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
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="SEASafeguardDataset", choices=list(AVAILABLE_DATASETS.keys()))
    parser.add_argument("--mixed_tasks_sample", action="store_true")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--cultural", type=str, default="th")
    parser.add_argument("--subset", type=str, default="general")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_logprobs", type=int, default=2)
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
    parser.add_argument("--sensitive_as_harmful", action="store_true")
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--prompt_intervention", action="store_true")
    parser.add_argument("--prompt_intervention_2", action="store_true")
    parser.add_argument("--lmi_label", type=str, default="Harmful", choices=["Harmful", "Safe"])
    parser.add_argument("--lmi_threshold", type=float, default=0.0)
    parser.add_argument("--lmi_k", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="./outputs/inference_safeguard.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # with open("./data/LMI_scores.jsonl", "r") as f:
    #     data = json.load(f)
    #     tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
    #     train_harmful_LMIs = [(k, v) for k, v in data["LMI"]["harmful"].items() if v > args.lmi_threshold][:args.lmi_k]
    #     train_harmful_LMIs = {int(k): v for k, v in train_harmful_LMIs}
    #     train_safe_LMIs = [(k, v) for k, v in data["LMI"]["benign"].items() if v > args.lmi_threshold][:args.lmi_k]
    #     train_safe_LMIs = {int(k): v for k, v in train_safe_LMIs}

    # # Get mask_tokens
    # mask_tokens = None
    # if args.mask_topk_tokens is not None:
    #     with open("./data/LMI_scores.jsonl", "r") as f:
    #         data = json.load(f)
    #         train_harmful_LMIs = list(data["LMI"]["harmful"])[:args.mask_topk_tokens]
    #         mask_tokens = [int(k) for k in train_harmful_LMIs]
    #     print(f"Mask tokens: {mask_tokens}")

    # # Report gpu availability
    # if torch.cuda.is_available():
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    #     print(f"GPU number of devices: {torch.cuda.device_count()}")
    # else:
    #     print("Using CPU")

    # Initial safeguard
    # print(f"Safeguard model: {args.safeguard_name}")
    safeguard = load_safeguard(
        safeguard_name=args.safeguard_name,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        api_key=args.api_key,
        api_base=args.api_base,
        disable_logitlens=args.disable_logitlens,
        top_logprobs=args.top_logprobs,
        max_tokens=args.max_tokens,
    )

    # Initial dataset
    dataset = load_dataset(
        args.dataset_name, 
        args.prompt_intention, 
        args.attack_engine, 
        args.language,
        cultural=args.cultural,
        subset=args.subset,
    )
    samples: List[SafeGuardInput] = dataset.as_samples(split=args.dataset_split)
    random.shuffle(samples)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    # Create save directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    prompt_unknown = 0
    prompt_harmful_scores = []
    prompt_gold_labels = []
    response_unknown = 0
    response_harmful_scores = []
    response_gold_labels = []
    with open(args.output_path, "w") as f:
        for sample in tqdm(samples):
            safeguard_response: SafeGuardResponse = safeguard.predict(input=sample)
            print(safeguard_response)
            if sample.response is not None:
                # Get response classification results
                response_harmful_score = [score for label, score, logit in safeguard_response.response_labels if label == "Harmful" or (args.sensitive_as_harmful and label == "Sensitive")]
                if len(response_harmful_score) == 0:
                    response_unknown += 1
                    response_harmful_score = 0.0
                else:
                    response_harmful_score = response_harmful_score[0]
                response_harmful_scores.append(response_harmful_score)
                response_gold_labels.append(int(sample.response_gold_label == "Harmful" or (args.sensitive_as_harmful and sample.response_gold_label == "Sensitive")))
                if not args.mixed_tasks_sample:
                    continue
            # Get prompt classification results
            prompt_harmful_score = [score for label, score, logit in safeguard_response.prompt_labels if label == "Harmful" or (args.sensitive_as_harmful and label == "Sensitive")]
            if len(prompt_harmful_score) == 0:
                prompt_unknown += 1
                prompt_harmful_score = 0.0
            else:
                prompt_harmful_score = prompt_harmful_score[0]
            prompt_harmful_scores.append(prompt_harmful_score)
            prompt_gold_labels.append(int(sample.prompt_gold_label == "Harmful" or (args.sensitive_as_harmful and sample.response_gold_label == "Sensitive")))
            # Save results
            f.write(json.dumps({
                "prompt": sample.prompt,
                "prompt_labels": safeguard_response.prompt_labels,
                "prompt_gold_label": sample.prompt_gold_label,
                "response": sample.response,
                "response_labels": safeguard_response.response_labels,
                "response_gold_label": sample.response_gold_label,
                "metadata": safeguard_response.metadata,
            }, ensure_ascii=False) + "\n")

    # Report Prompt Classification Performance
    assert len(prompt_gold_labels) > 0 and len(prompt_harmful_scores) > 0, "No prompt samples, make sure that the mixed_tasks_sample is set correctly."
    metrics = SafeGuardMetric()
    metrics.update(prompt_gold_labels, prompt_harmful_scores)
    print("Prompt Classification Performance:")
    print(f"Recall: {round(metrics.recall * 100, 1)}")
    print(f"Precision: {round(metrics.precision * 100, 1)}")
    print(f"F1: {round(metrics.f1 * 100, 1)}")
    print(f"AUPRC: {round(metrics.pr_auc * 100, 1)}")
    print(f"Unknown label prediction: {prompt_unknown}")

    # Report Response Classification Performance
    metrics = SafeGuardMetric()
    metrics.update(response_gold_labels, response_harmful_scores)
    print("Response Classification Performance:")
    print(f"Recall: {round(metrics.recall * 100, 1)}")
    print(f"Precision: {round(metrics.precision * 100, 1)}")
    print(f"F1: {round(metrics.f1 * 100, 1)}")
    print(f"AUPRC: {round(metrics.pr_auc * 100, 1)}")
    print(f"Unknown label prediction: {response_unknown}")