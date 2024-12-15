import json
import argparse
from tqdm import tqdm
from iris.metrics.safeguard_metrics import SafeGuardMetric


"""
python scripts/eval_record.py \
--load_path_a1 ./outputs/wildguard/ORBenchDataset/test/hard_benign_prompts.jsonl \
--load_path_a2 ./outputs/wildguard/ORBenchDataset/test/harmful_prompts.jsonl \
--dataset_name ORBench

python scripts/eval_record.py \
--load_path_a1 ./outputs/wildguard/ORBenchDataset/test/hard_benign_prompts.jsonl \
--load_path_a2 ./outputs/wildguard/ORBenchDataset/test/harmful_prompts.jsonl \
--load_path_b1 ./outputs/wildguard/ORBenchDataset/test/mask_prefix/hard_benign_prompts.jsonl \
--load_path_b2 ./outputs/wildguard/ORBenchDataset/test/mask_prefix/harmful_prompts.jsonl \
--dataset_name ORBench

python scripts/eval_record.py \
--load_path_a1 ./outputs/wildguard/WildGuardMixDataset/train/4000_prompts.jsonl \
--dataset_name WildGuardMix
"""


def load_examples(load_path: str):
    gold_labels = []
    pred_scores = []
    with open(load_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            gold_label = int(example["label"] == "Harmful")
            if len(example["response"][0]) == 2:
                pred_score = [score for label, score in example["response"] if label == "Harmful"][0]
            else:
                pred_score = [score for label, score, _ in example["response"] if label == "Harmful"][0]
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
    print((metrics_a.precision, metrics_a.recall, metrics_a.f1))
    # metrics_a.plot(args.dataset_name)

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
    print((metrics_b.precision, metrics_b.recall, metrics_b.f1))

    metrics_b.plot_comparison(metrics_a, "WildGuard", args.dataset_name)