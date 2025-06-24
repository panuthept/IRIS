import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from iris.metrics.safeguard_metrics import SafeGuardMetric


def get_gold_label(label: str, sensitive_as_harmful: bool = True) -> int:
    """ Convert label to 0 or 1 """
    label = label.lower()
    if label in ["harmful", "unsafe"]:
        return 1
    elif label in ["safe", "unharmful"]:
        return 0
    elif label == "sensitive":
        return int(sensitive_as_harmful)
    else:
        raise ValueError(f"Unknown label: {label}")
    
def get_pred_score(labels: List[Tuple[str, float, float]], sensitive_as_harmful: bool = True) -> float:
    harmful_score = 0.0
    # print(labels)
    for label, score, _ in labels:
        label = label.lower()
        if label == "harmful":
            harmful_score += score
        elif label == "sensitive" and sensitive_as_harmful:
            harmful_score += score
    return harmful_score

def get_score_distribution(result_path: str, sensitive_as_harmful: bool = True):
    label_to_id = {
        "Safe": 0,
        "Sensitive": 1,
        "Harmful": 2
    }
    prompt_xs = []
    prompt_ys = []
    response_xs = []
    response_ys = []
    prompt_scores = [[], [], []]
    response_scores = [[], [], []]
    with open(result_path, "r") as f:
        for line in f:
            example = json.loads(line)
            x = label_to_id[example["prompt_gold_label"]]
            prompt_xs.append(x)
            y = get_pred_score(example["prompt_labels"], sensitive_as_harmful)
            # prompt_ys.append(map[example["prompt_gold_label"]])
            prompt_ys.append(y / 0.5)
            prompt_scores[x].append(y)
            if example["response"] is not None:
                x = label_to_id[example["response_gold_label"]]
                response_xs.append(x)
                y = get_pred_score(example["response_labels"], sensitive_as_harmful)
                response_ys.append(y / 0.5)
                response_scores[x].append(y)

    # Set the figure title
    print(result_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for i, label in enumerate(["Safe", "Sensitive", "Harmful"]):
        if len(prompt_scores[i]) == 0:
            continue
        mean_score = np.mean(prompt_scores[i])
        std_score = np.std(prompt_scores[i])
        print(f"{mean_score} ± {std_score} ({label})")
    print("-" * 100)
    # m, b = np.polyfit(prompt_xs, prompt_ys, 1)
    # print((m, b))
    plt.boxplot(prompt_scores)
    plt.ylabel("Score")
    plt.xlabel("Label")
    plt.xticks([1, 2, 3], ["Safe", "Sensitive", "Harmful"])
    plt.title(f"Prompt Classification")

    plt.subplot(1, 2, 2)
    for i, label in enumerate(["Safe", "Sensitive", "Harmful"]):
        if len(response_scores[i]) == 0:
            continue
        mean_score = np.mean(response_scores[i])
        std_score = np.std(response_scores[i])
        print(f"{mean_score} ± {std_score} ({label})")
    print("=" * 100)
    # m, b = np.polyfit(response_xs, response_ys, 1)
    # print((m, b))
    plt.boxplot(response_scores)
    plt.ylabel("Score")
    plt.xlabel("Label")
    plt.xticks([1, 2, 3], ["Safe", "Sensitive", "Harmful"])
    plt.title(f"Response Classification")
    plt.show()

def get_eval_results(result_path: str, sensitive_as_harmful: bool = True, ignore_sensitive: bool = False) -> Dict[str, List[float]]:
    """ Return results in the format of tuple (F1, AUPRC, FPR) for each task (prompt_clf, response_clf) """
    prompt_clf_preds = []
    prompt_clf_labels = []
    response_clf_preds = []
    response_clf_labels = []
    with open(result_path, "r") as f:
        for line in f:
            example = json.loads(line)
            if not ignore_sensitive or (ignore_sensitive and example["prompt_gold_label"].lower() != "sensitive"):
                prompt_clf_preds.append(get_pred_score(example["prompt_labels"], sensitive_as_harmful))
                prompt_clf_labels.append(get_gold_label(example["prompt_gold_label"], sensitive_as_harmful))
            if example["response_gold_label"] is not None:
                if not ignore_sensitive or (ignore_sensitive and example["response_gold_label"].lower() != "sensitive"):
                    response_clf_preds.append(get_pred_score(example["response_labels"], sensitive_as_harmful))
                    response_clf_labels.append(get_gold_label(example["response_gold_label"], sensitive_as_harmful))
    results = {}
    if len(prompt_clf_preds) == 0:
        print(result_path)
        results["Prompt_clf"] = [None, None, None]
        results["Response_clf"] = [None, None, None]
        return results

    # Get Prompt Classification Performance
    metrics = SafeGuardMetric()
    metrics.update(prompt_clf_labels, prompt_clf_preds)
    results["Prompt_clf"] = [round(metrics.best_f1 * 100, 1), round(metrics.pr_auc * 100, 1), round(metrics.best_fpr * 100, 1)]
    # metrics.plot(dataset_name=f"Prompt Classification ({result_path})")

    # Report Response Classification Performance
    if len(response_clf_labels) > 0 and len(response_clf_preds) > 0:
        metrics = SafeGuardMetric()
        metrics.update(response_clf_labels, response_clf_preds)
        results["Response_clf"] = [round(metrics.best_f1 * 100, 1), round(metrics.pr_auc * 100, 1), round(metrics.best_fpr * 100, 1)]
    else:
        results["Response_clf"] = [None, None, None]
    # metrics.plot(dataset_name=f"Response Classification ({result_path})")
    return results

def report_results(paths, ignore_sensitive: bool = True, sensitive_as_harmful: bool = True):
    prompt_clf_results = []
    response_clf_results = []
    for path in paths:
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful, ignore_sensitive)
            prompt_clf_results.extend(results["Prompt_clf"])
            response_clf_results.extend(results["Response_clf"])
        else:
            prompt_clf_results.extend([None, None, None])
            response_clf_results.extend([None, None, None])
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*3] for i in range(len(paths)) if prompt_clf_results[i*3] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*3+1] for i in range(len(paths)) if prompt_clf_results[i*3+1] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*3+2] for i in range(len(paths)) if prompt_clf_results[i*3+2] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*3] for i in range(len(paths)) if response_clf_results[i*3] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*3+1] for i in range(len(paths)) if response_clf_results[i*3+1] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*3+2] for i in range(len(paths)) if response_clf_results[i*3+2] is not None]), 1))

    print("Prompt CLF:")
    print(" & ".join([str(v) for v in prompt_clf_results]))
    print("Response CLF:")
    print(" & ".join([str(v) for v in response_clf_results]))

def report_sum_results(paths):
    prompt_clf_results = []
    response_clf_results = []
    # Get General Set Performance (English)
    for path in paths["general"][:1]:
        if os.path.exists(path):
            results = get_eval_results(path)
            prompt_clf_results.extend(results["Prompt_clf"])
            response_clf_results.extend(results["Response_clf"])
        else:
            prompt_clf_results.extend([None, None, None])
            response_clf_results.extend([None, None, None])
    # Get General Set Performance (SEA)
    c = 0
    t_prompt_clf_results = []
    t_response_clf_results = []
    for path in paths["general"][1:]:
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3] for i in range(c) if t_prompt_clf_results[i*3] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+1] for i in range(c) if t_prompt_clf_results[i*3+1] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+2] for i in range(c) if t_prompt_clf_results[i*3+2] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3] for i in range(c) if t_response_clf_results[i*3] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+1] for i in range(c) if t_response_clf_results[i*3+1] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+2] for i in range(c) if t_response_clf_results[i*3+2] is not None]), 1))
    prompt_clf_results.extend(t_prompt_clf_results[-3:])
    response_clf_results.extend(t_response_clf_results[-3:])
    # Get Synthetic Cultural Set Performance (English)
    c = 0
    t_prompt_clf_results = []
    t_response_clf_results = []
    for path in paths["cultural_en"]:
        # No-Sensitive
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=False, ignore_sensitive=True)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
        # Sensitive-as-Harmful
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=True, ignore_sensitive=False)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
        # Sensitive-as-Safe
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=False, ignore_sensitive=False)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3] for i in range(c) if t_prompt_clf_results[i*3] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+1] for i in range(c) if t_prompt_clf_results[i*3+1] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+2] for i in range(c) if t_prompt_clf_results[i*3+2] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3] for i in range(c) if t_response_clf_results[i*3] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+1] for i in range(c) if t_response_clf_results[i*3+1] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+2] for i in range(c) if t_response_clf_results[i*3+2] is not None]), 1))
    prompt_clf_results.extend(t_prompt_clf_results[-3:])
    response_clf_results.extend(t_response_clf_results[-3:])
    # Get Synthetic Cultural Set Performance (SEA)
    c = 0
    t_prompt_clf_results = []
    t_response_clf_results = []
    for path in paths["cultural_local"][1:]:    # Ignore English (Global culture)
        # No-Sensitive
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=False, ignore_sensitive=True)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
        # Sensitive-as-Harmful
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=True, ignore_sensitive=False)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
        # Sensitive-as-Safe
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path, sensitive_as_harmful=False, ignore_sensitive=False)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3] for i in range(c) if t_prompt_clf_results[i*3] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+1] for i in range(c) if t_prompt_clf_results[i*3+1] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+2] for i in range(c) if t_prompt_clf_results[i*3+2] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3] for i in range(c) if t_response_clf_results[i*3] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+1] for i in range(c) if t_response_clf_results[i*3+1] is not None]), 1))
    t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+2] for i in range(c) if t_response_clf_results[i*3+2] is not None]), 1))
    prompt_clf_results.extend(t_prompt_clf_results[-3:])
    response_clf_results.extend(t_response_clf_results[-3:])
    # Get Hand-Written Cultural Set Performance (English)
    c = 0
    t_prompt_clf_results = []
    t_response_clf_results = []
    for path in paths["cultural_handwritten_en"]:
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3] for i in range(c) if t_prompt_clf_results[i*3] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+1] for i in range(c) if t_prompt_clf_results[i*3+1] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+2] for i in range(c) if t_prompt_clf_results[i*3+2] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3] for i in range(c) if t_response_clf_results[i*3] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+1] for i in range(c) if t_response_clf_results[i*3+1] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+2] for i in range(c) if t_response_clf_results[i*3+2] is not None]), 1))
    prompt_clf_results.extend(t_prompt_clf_results[-3:])
    # response_clf_results.extend(t_response_clf_results[-3:])
    # Get Hand-Written Cultural Set Performance (Native)
    c = 0
    t_prompt_clf_results = []
    t_response_clf_results = []
    for path in paths["cultural_handwritten_local"]:
        c += 1
        if os.path.exists(path):
            results = get_eval_results(path)
            t_prompt_clf_results.extend(results["Prompt_clf"])
            t_response_clf_results.extend(results["Response_clf"])
        else:
            t_prompt_clf_results.extend([None, None, None])
            t_response_clf_results.extend([None, None, None])
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3] for i in range(c) if t_prompt_clf_results[i*3] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+1] for i in range(c) if t_prompt_clf_results[i*3+1] is not None]), 1))
    t_prompt_clf_results.append(round(np.mean([t_prompt_clf_results[i*3+2] for i in range(c) if t_prompt_clf_results[i*3+2] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3] for i in range(c) if t_response_clf_results[i*3] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+1] for i in range(c) if t_response_clf_results[i*3+1] is not None]), 1))
    # t_response_clf_results.append(round(np.mean([t_response_clf_results[i*3+2] for i in range(c) if t_response_clf_results[i*3+2] is not None]), 1))
    prompt_clf_results.extend(t_prompt_clf_results[-3:])
    # response_clf_results.extend(t_response_clf_results[-3:])

    # Final average (English)
    c = 3
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6] for i in range(c) if prompt_clf_results[i*6] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6+1] for i in range(c) if prompt_clf_results[i*6+1] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6+2] for i in range(c) if prompt_clf_results[i*6+2] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6+3] for i in range(c) if prompt_clf_results[i*6+3] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6+1+3] for i in range(c) if prompt_clf_results[i*6+1+3] is not None]), 1))
    prompt_clf_results.append(round(np.mean([prompt_clf_results[i*6+2+3] for i in range(c) if prompt_clf_results[i*6+2+3] is not None]), 1))
    c = 2
    response_clf_results.append(round(np.mean([response_clf_results[i*6] for i in range(c) if response_clf_results[i*6] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*6+1] for i in range(c) if response_clf_results[i*6+1] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*6+2] for i in range(c) if response_clf_results[i*6+2] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*6+3] for i in range(c) if response_clf_results[i*6+3] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*6+1+3] for i in range(c) if response_clf_results[i*6+1+3] is not None]), 1))
    response_clf_results.append(round(np.mean([response_clf_results[i*6+2+3] for i in range(c) if response_clf_results[i*6+2+3] is not None]), 1))
    
    print("Prompt CLF:")
    print(" & ".join([str(v) for v in prompt_clf_results]))
    print("Response CLF:")
    print(" & ".join([str(v) for v in response_clf_results]))

if __name__ == "__main__":
    ignore_sensitive = False
    sensitive_as_harmful = False

    # base_path = "./outputs/outputs/LLMGuard-Gemma3-4B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LLMGuard-Gemma3-27B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LLMGuard-Llama3.1-8B/SEASafeguardDataset"
    base_path = "./outputs/outputs/LLMGuard-Llama3.1-70B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LLMGuard-Llama3.2-3B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LLMGuard-Llama3.3-70B/SEASafeguardDataset"

    # base_path = "./outputs/outputs/ShieldGemma-2B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/ShieldGemma/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LlamaGuard-1B/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LlamaGuard/SEASafeguardDataset"
    # base_path = "./outputs/outputs/LlamaGuard4/SEASafeguardDataset"
    # base_path = "./outputs/outputs/PolyGuard-Qwen-Smol/SEASafeguardDataset"
    # base_path = "./outputs/outputs/PolyGuard-Qwen/SEASafeguardDataset"
    # base_path = "./outputs/outputs/PolyGuard-Ministral/SEASafeguardDataset"

    # base_path = "./outputs/outputs/Llama-Guard-v3-N-S/SEASafeguardDataset"
    
    general_results = {"Prompt_CLF": [], "Response_CLF": []}
    cultural_en_results = {"Prompt_CLF": [], "Response_CLF": []}
    cultural_local_results = {"Prompt_CLF": [], "Response_CLF": []}
    cultural_handwritten_en_results = {"Prompt_CLF": [], "Response_CLF": []}
    cultural_handwritten_local_results = {"Prompt_CLF": [], "Response_CLF": []}

    paths = {
        "general": [
            os.path.join(base_path, "general", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "ta", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "th", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "tl", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "ms", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "in", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "my", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "general", "vi", "test", "all_prompts.jsonl"),
        ],
        "cultural_en": [
            os.path.join(base_path, "en_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ta_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "th_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "tl_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ms_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "in_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "my_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "vi_cultural", "en", "test", "all_prompts.jsonl"),
        ],
        "cultural_local": [
            os.path.join(base_path, "en_cultural", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ta_cultural", "ta", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "th_cultural", "th", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "tl_cultural", "tl", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ms_cultural", "ms", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "in_cultural", "in", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "my_cultural", "my", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "vi_cultural", "vi", "test", "all_prompts.jsonl"),
        ],
        "cultural_handwritten_en": [
            os.path.join(base_path, "ta_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "th_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "tl_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ms_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "in_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "my_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "vi_cultural_handwritten", "en", "test", "all_prompts.jsonl"),
        ],
        "cultural_handwritten_local": [
            os.path.join(base_path, "ta_cultural_handwritten", "ta", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "th_cultural_handwritten", "th", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "tl_cultural_handwritten", "tl", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "ms_cultural_handwritten", "ms", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "in_cultural_handwritten", "in", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "my_cultural_handwritten", "my", "test", "all_prompts.jsonl"),
            os.path.join(base_path, "vi_cultural_handwritten", "vi", "test", "all_prompts.jsonl"),
        ]
    }

    # # Get General Set Performance
    # print("General Results:")
    # report_results(paths["general"], ignore_sensitive=ignore_sensitive, sensitive_as_harmful=sensitive_as_harmful)

    # # Get Cultural Specific (English) Set Performance
    # print("Synthetic Cultural Results (English):")
    # report_results(paths["cultural_en"], ignore_sensitive=ignore_sensitive, sensitive_as_harmful=sensitive_as_harmful)

    # # Get Cultural Specific (Native) Set Performance
    # print("Synthetic Cultural Results (Native):")
    # report_results(paths["cultural_local"][1:], ignore_sensitive=ignore_sensitive, sensitive_as_harmful=sensitive_as_harmful)

    # # Get Hand-Written Cultural Specific (English) Set Performance
    # print("Hand-Written Cultural Results (English):")
    # report_results(paths["cultural_handwritten_en"], ignore_sensitive=ignore_sensitive, sensitive_as_harmful=sensitive_as_harmful)

    # # Get Hand-Written Cultural Specific (Native) Set Performance
    # print("Hand-Written Cultural Results (Native):")
    # report_results(paths["cultural_handwritten_local"], ignore_sensitive=ignore_sensitive, sensitive_as_harmful=sensitive_as_harmful)

    # Get Summary Results
    print("Summary Results:")
    report_sum_results(paths)