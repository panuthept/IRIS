import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class Metrics:
    def __init__(self):
        self.tpr = None
        self.fpr = None
        self.threshold = None
        self.auc = None

    def update(self, gold_labels, pred_scores):
        self.fpr, self.tpr, self.threshold = roc_curve(gold_labels, pred_scores)
        self.auc = np.trapz(self.tpr, self.fpr)

        # self.harmful_count = np.sum(gold_labels)
        # self.benign_count = len(gold_labels) - self.harmful_count

        # if gold_label == "Harmful":
        #     self.harmful_count += 1
        #     if response[0][0] == "Harmful":
        #         self.tp += 1
        # else:
        #     self.benign_count += 1
        #     if response[0][0] == "Harmful":
        #         self.fp += 1

    def get_tpr(self):
        return self.tp / (self.harmful_count + 1e-7)

    def get_fpr(self):
        return self.fp / (self.benign_count + 1e-7)


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
    return np.array(gold_labels), np.array(pred_scores)


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
        all_gold_labels = np.concatenate([all_gold_labels, gold_labels])
        all_pred_scores = np.concatenate([all_pred_scores, pred_scores])
    if args.load_path_a3:
        gold_labels, pred_scores = load_examples(args.load_path_a3)
        all_gold_labels = np.concatenate([all_gold_labels, gold_labels])
        all_pred_scores = np.concatenate([all_pred_scores, pred_scores])

    metrics_a = Metrics()
    metrics_a.update(all_gold_labels, all_pred_scores)
    print(f"AUC: {round(metrics_a.auc, 2)}")
    print([(tpr, fpr, thr) for tpr, fpr, thr in zip(metrics_a.tpr, metrics_a.fpr, metrics_a.threshold)])
    # print(f"TPR: {round([tpr for tpr, thr in zip(metrics_a.tpr, metrics_a.threshold) if thr == 0.5][0], 2)}")
    # print(f"FPR: {round([fpr for fpr, thr in zip(metrics_a.fpr, metrics_a.threshold) if thr == 0.5][0], 2)}")

    all_gold_labels, all_pred_scores = load_examples(args.load_path_b1)
    if args.load_path_b2:
        gold_labels, pred_scores = load_examples(args.load_path_b2)
        all_gold_labels = np.concatenate([all_gold_labels, gold_labels])
        all_pred_scores = np.concatenate([all_pred_scores, pred_scores])
    if args.load_path_b3:
        gold_labels, pred_scores = load_examples(args.load_path_b3)
        all_gold_labels = np.concatenate([all_gold_labels, gold_labels])
        all_pred_scores = np.concatenate([all_pred_scores, pred_scores])

    metrics_b = Metrics()
    metrics_b.update(all_gold_labels, all_pred_scores)
    print(f"AUC: {round(metrics_b.auc, 2)}")
    print([(tpr, fpr, thr) for tpr, fpr, thr in zip(metrics_b.tpr, metrics_b.fpr, metrics_b.threshold)])

    plt.plot(metrics_a.fpr, metrics_a.tpr, color='red')
    plt.plot(metrics_b.fpr, metrics_b.tpr, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    # print(f"TPR: {round(metrics.get_tpr(), 2)}")
    # print(f"FPR: {round(metrics.get_fpr(), 2)}")