import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, 
    precision_recall_curve, 
    roc_curve, 
    auc
)


class SafeGuardMetric:
    def __init__(self):
        self.tprs = []
        self.fprs = []
        self.roc_thresholds = []
        self.roc_auc = None
        self.recalls = []
        self.precisions = []
        self.pr_thresholds = []
        self.pr_auc = None
        self.f1s = []
        self.recall = None
        self.precision = None
        self.f1 = None
        self.fpr = None

    def update(self, gold_labels, pred_scores):
        """
        gold_labels: List[int], 1 for harmful, 0 for benign
        pred_scores: List[float], the confident scores for harmful
        """
        # Convert to NumpyArray
        gold_labels = np.array(gold_labels)
        pred_scores = np.array(pred_scores)
        # Compute FPR
        self.fpr = np.sum((pred_scores > 0.5) & (gold_labels == 0)) / np.sum(gold_labels == 0)
        # Compute Recall, Precision, F1
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(gold_labels, pred_scores > 0.5, average='binary')
        # Compute ROC curve and ROC area
        self.fprs, self.tprs, self.roc_thresholds = roc_curve(gold_labels, pred_scores)
        self.precisions, self.recalls, self.pr_thresholds = precision_recall_curve(gold_labels, pred_scores)
        self.pr_thresholds = np.append(self.pr_thresholds, 1)
        self.roc_auc = auc(self.fprs, self.tprs)
        self.pr_auc = auc(self.recalls, self.precisions)
        # Compute F1 score
        self.f1s = 2 * self.precisions * self.recalls / (self.precisions + self.recalls + 1e-7)

    def plot(
            self, 
            dataset_name: str = None, 
            figsize=(13, 5),
            show_pr_curve=True,
            table_name: str = None
    ):
        plt.figure(figsize=figsize)
        if show_pr_curve:
            plt.subplot(1, 2, 1)
            plt.plot(self.recalls, self.precisions, color='darkorange', lw=1, label='AUC = %0.2f' % (self.pr_auc * 100))
            # plt.plot(self.precisions, self.recalls, color='darkorange', lw=1, label='AUC = %0.2f' % (self.pr_auc * 100))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curve')
            plt.legend(loc="lower right")

            plt.subplot(1, 2, 2)
        plt.plot(self.pr_thresholds, self.recalls, color='blue', lw=1, label='Recall')
        plt.plot(self.pr_thresholds, self.precisions, color='green', lw=1, label='Precision')
        plt.plot(self.pr_thresholds, self.f1s, color='red', lw=1, label='F1')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('Performance')
        plt.title(table_name if table_name else 'Threshold Curve')
        plt.legend(loc="lower right")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()

    def plot_comparison(self, other_metrics: 'SafeGuardMetric', other_name: str, dataset_name: str = None):
        plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        plt.plot(other_metrics.recalls, other_metrics.precisions, linestyle='dashed', color='darkorange', lw=1, label='AUC = %0.2f (%s)' % (other_metrics.pr_auc * 100, other_name))
        plt.plot(self.recalls, self.precisions, color='darkorange', lw=1, label='AUC = %0.2f (Our)' % (self.pr_auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(other_metrics.pr_thresholds, other_metrics.f1s, linestyle='dashed', color='red', lw=1, label='F1 (%s)' % other_name)
        plt.plot(other_metrics.pr_thresholds, other_metrics.recalls, linestyle='dashed', color='blue', lw=1, label='Recall (%s)' % other_name)
        plt.plot(other_metrics.pr_thresholds, other_metrics.precisions, linestyle='dashed', color='green', lw=1, label='Precision (%s)' % other_name)
        plt.plot(self.pr_thresholds, self.f1s, color='red', lw=1, label='F1 (Our)')
        plt.plot(self.pr_thresholds, self.recalls, color='blue', lw=1, label='Recall (Our)')
        plt.plot(self.pr_thresholds, self.precisions, color='green', lw=1, label='Precision (Our)')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Curve')
        plt.legend(loc="lower right")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()