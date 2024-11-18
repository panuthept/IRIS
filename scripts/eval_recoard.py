import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from iris.metrics.safeguard_metrics import SafeGuardMetric


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

    # best_index = np.argmax(metrics_a.f1s)
    # best_f1_a = metrics_a.f1s[best_index]
    # best_recall_a = metrics_a.recalls[best_index]
    # best_precision_a = metrics_a.precisions[best_index]

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

    metrics_b.plot_comparison(metrics_a, "WildGuard", "JailbreakBench Dataset")

    # best_index = np.argmax(metrics_b.f1s)
    # best_f1_b = metrics_b.f1s[best_index]
    # best_recall_b = metrics_b.recalls[best_index]
    # best_precision_b = metrics_b.precisions[best_index]

    # # Comparison plots
    # plt.subplot(1, 2, 1)
    # plt.plot(metrics_a.fprs, metrics_a.tprs, linestyle='dashed', color='darkorange', lw=1, label='AUC = %0.2f (WildGuard)' % (metrics_a.auc * 100))
    # plt.plot(metrics_b.fprs, metrics_b.tprs, color='darkorange', lw=1, label='AUC = %0.2f (Our)' % (metrics_b.auc * 100))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")

    # plt.subplot(1, 2, 2)
    # plt.plot(metrics_a.thresholds, metrics_a.f1s, linestyle='dashed', color='red', lw=1, label='F1 = %0.2f (WildGuard)' % (best_f1_a * 100))
    # plt.plot(metrics_a.thresholds, metrics_a.recalls, linestyle='dashed', color='blue', lw=1, label='Recall = %0.2f (WildGuard)' % (best_recall_a * 100))
    # plt.plot(metrics_a.thresholds, metrics_a.precisions, linestyle='dashed', color='green', lw=1, label='Precision = %0.2f (WildGuard)' % (best_precision_a * 100))
    # plt.plot(metrics_b.thresholds, metrics_b.f1s, color='red', lw=1, label='F1 = %0.2f (Our)' % (best_f1_b * 100))
    # plt.plot(metrics_b.thresholds, metrics_b.recalls, color='blue', lw=1, label='Recall = %0.2f (Our)' % (best_recall_b * 100))
    # plt.plot(metrics_b.thresholds, metrics_b.precisions, color='green', lw=1, label='Precision = %0.2f (Our)' % (best_precision_b * 100))
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Threshold')
    # plt.ylabel('F1, Precision, Recall')
    # plt.title('Performance')
    # plt.legend(loc="lower left")

    # # Add figure title
    # plt.suptitle("JailbreakBench Dataset")
    # plt.show()