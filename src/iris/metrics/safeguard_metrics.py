import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


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

    def update(self, gold_labels, pred_scores):
        """
        gold_labels: List[int], 1 for harmful, 0 for benign
        pred_scores: List[float], the confident scores for harmful
        """
        # Convert to NumpyArray
        gold_labels = np.array(gold_labels)
        pred_scores = np.array(pred_scores)
        # Compute ROC curve and ROC area
        self.fprs, self.tprs, self.roc_thresholds = roc_curve(gold_labels, pred_scores)
        self.precisions, self.recalls, self.pr_thresholds = precision_recall_curve(gold_labels, pred_scores)
        self.roc_auc = auc(self.fprs, self.tprs)
        self.pr_auc = auc(self.recalls, self.precisions)
        # Compute F1 score
        self.f1s = 2 * self.precisions * self.recalls / (self.precisions + self.recalls + 1e-7)

    def plot(self, dataset_name: str = None):
        plt.subplot(1, 2, 1)
        plt.plot(self.fprs, self.tprs, color='blue', lw=1, label='AUC = %0.2f' % (self.roc_auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(self.recalls, self.precisions, color='blue', lw=1, label='AUC = %0.2f' % (self.pr_auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()

    def plot_comparison(self, other_metrics: 'SafeGuardMetric', other_name: str, dataset_name: str = None):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(other_metrics.fprs, other_metrics.tprs, color='red', lw=1, label='AUC = %0.2f (%s)' % (other_metrics.roc_auc * 100, other_name))
        plt.plot(self.fprs, self.tprs, color='blue', lw=1, label='AUC = %0.2f (Our)' % (self.roc_auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(other_metrics.recalls, other_metrics.precisions, color='red', lw=1, label='AUC = %0.2f (%s)' % (other_metrics.pr_auc * 100, other_name))
        plt.plot(self.recalls, self.precisions, color='blue', lw=1, label='AUC = %0.2f (Our)' % (self.pr_auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()