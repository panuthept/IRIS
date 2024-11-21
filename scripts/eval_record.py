import json
import argparse
from tqdm import tqdm
from iris.metrics.safeguard_metrics import SafeGuardMetric


"""
python scripts/eval_record.py \
--load_path_a1 ./outputs/wildguard/ORBenchDataset/test/hard_benign_prompts.jsonl \
--load_path_a2 ./outputs/wildguard/ORBenchDataset/test/harmful_prompts.jsonl \
--load_path_b1 ./outputs/iris_wildguard_layer_19/ORBenchDataset/test/hard_benign_prompts.jsonl \
--load_path_b2 ./outputs/iris_wildguard_layer_19/ORBenchDataset/test/harmful_prompts.jsonl \
--dataset_name ORBench
"""


def load_examples(load_path: str):
    gold_labels = []
    pred_scores = []
    with open(load_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            gold_label = int(example["label"] == "Harmful")
            pred_score = [score for label, score in example["response"] if label == "Harmful"][0]
            gold_labels.append(gold_label)
            pred_scores.append(pred_score)
    return gold_labels, pred_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path_a1", type=str, required=True)
    parser.add_argument("--load_path_a2", type=str, default=None)
    parser.add_argument("--load_path_a3", type=str, default=None)
    parser.add_argument("--load_path_b1", type=str, required=True)
    parser.add_argument("--load_path_b2", type=str, default=None)
    parser.add_argument("--load_path_b3", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    all_gold_labels, all_pred_scores = load_examples(args.load_path_a1)
    if args.load_path_a2:
        gold_labels, pred_scores = load_examples(args.load_path_a2)
        all_gold_labels.extend(gold_labels)
        all_pred_scores.extend(pred_scores)
    if args.load_path_a3:
        gold_labels, pred_scores = load_examples(args.load_path_a3)
        all_gold_labels.extend(gold_labels)
        all_pred_scores.extend(pred_scores)

    metrics_a = SafeGuardMetric()
    metrics_a.update(all_gold_labels, all_pred_scores)

    all_gold_labels, all_pred_scores = load_examples(args.load_path_b1)
    if args.load_path_b2:
        gold_labels, pred_scores = load_examples(args.load_path_b2)
        all_gold_labels.extend(gold_labels)
        all_pred_scores.extend(pred_scores)
    if args.load_path_b3:
        gold_labels, pred_scores = load_examples(args.load_path_b3)
        all_gold_labels.extend(gold_labels)
        all_pred_scores.extend(pred_scores)

    metrics_b = SafeGuardMetric()
    metrics_b.update(all_gold_labels, all_pred_scores)

    metrics_b.plot_comparison(metrics_a, "WildGuard", args.dataset_name)