import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class SafeGuardMetric:
    def __init__(self):
        self.tprs = []
        self.fprs = []
        self.recalls = []
        self.precisions = []
        self.f1s = []
        self.thresholds = []
        self.auc = None

    def update(self, gold_labels, pred_scores):
        """
        gold_labels: List[int], 1 for harmful, 0 for benign
        pred_scores: List[float], the confident scores for harmful
        """
        # Convert to NumpyArray
        gold_labels = np.array(gold_labels)
        pred_scores = np.array(pred_scores)
        # Compute ROC curve and ROC area
        self.fprs, self.tprs, self.thresholds = roc_curve(gold_labels, pred_scores)
        self.auc = auc(self.fprs, self.tprs)
        print(self.fprs)
        # Compute F1 score
        self.recalls = self.tprs
        self.precisions = np.where(self.tprs + self.fprs > 0, self.tprs / (self.tprs + self.fprs), 0)
        self.f1s = 2 * self.precisions * self.recalls / (self.precisions + self.recalls + 1e-7)

    def plot(self, dataset_name: str = None):
        plt.subplot(1, 2, 1)
        plt.plot(self.fprs, self.tprs, color='darkorange', lw=1, label='AUC = %0.2f' % (self.auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        best_index = np.argmax(self.f1s)
        best_f1 = self.f1s[best_index]
        best_recall = self.recalls[best_index]
        best_precision = self.precisions[best_index]

        plt.subplot(1, 2, 2)
        plt.plot(self.thresholds, self.f1s, color='red', lw=1, label='Best F1 = %0.2f' % (best_f1 * 100))
        plt.plot(self.thresholds, self.recalls, color='blue', lw=1, label='Best Recall = %0.2f' % (best_recall * 100))
        plt.plot(self.thresholds, self.precisions, color='green', lw=1, label='Best Precision = %0.2f' % (best_precision * 100))
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1, Precision, Recall')
        plt.title('Performance')
        plt.legend(loc="lower right")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()

    def plot_comparison(self, other_metrics: 'SafeGuardMetric', other_name: str, dataset_name: str = None):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(other_metrics.fprs, other_metrics.tprs, linestyle='dashed', color='darkorange', lw=1, label='AUC = %0.2f (%s)' % (other_metrics.auc * 100, other_name))
        plt.plot(self.fprs, self.tprs, color='darkorange', lw=1, label='AUC = %0.2f (Our)' % (self.auc * 100))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        best_index = np.argmax(self.f1s)
        best_f1 = self.f1s[best_index]
        best_recall = self.recalls[best_index]
        best_precision = self.precisions[best_index]

        best_index = np.argmax(other_metrics.f1s)
        other_best_f1 = other_metrics.f1s[best_index]
        other_best_recall = other_metrics.recalls[best_index]
        other_best_precision = other_metrics.precisions[best_index]

        plt.subplot(1, 2, 2)
        plt.plot(other_metrics.thresholds, other_metrics.f1s, linestyle='dashed', color='red', lw=1, label='Best F1 = %0.2f (%s)' % (other_best_f1 * 100, other_name))
        plt.plot(other_metrics.thresholds, other_metrics.recalls, linestyle='dashed', color='blue', lw=1, label='Best Recall = %0.2f (%s)' % (other_best_recall * 100, other_name))
        plt.plot(other_metrics.thresholds, other_metrics.precisions, linestyle='dashed', color='green', lw=1, label='Best Precision = %0.2f (%s)' % (other_best_precision * 100, other_name))
        plt.plot(self.thresholds, self.f1s, color='red', lw=1, label='Best F1 = %0.2f (Our)' % (best_f1 * 100))
        plt.plot(self.thresholds, self.recalls, color='blue', lw=1, label='Best Recall = %0.2f (Our)' % (best_recall * 100))
        plt.plot(self.thresholds, self.precisions, color='green', lw=1, label='Best Precision = %0.2f (Our)' % (best_precision * 100))
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1, Precision, Recall')
        plt.title('Performance')
        plt.legend(loc="lower left")

        if dataset_name:
            plt.suptitle(dataset_name)
        plt.show()

        


    # def plot_roc_curve(self):
    #     plt.figure()
    #     plt.plot(self.fprs, self.tprs, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % self.auc)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()