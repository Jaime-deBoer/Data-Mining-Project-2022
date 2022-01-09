"""
@author: Jaime de Boer
"""
import numpy as np


class PerformanceMetrics:
    """

    """
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp, self.fp, self.fn, self.tn = self.confusion_values()

    def confusion_values(self):
        """
        Calculate values that would be placed in a Confusion Matrix
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(self.y_pred)):
            if self.y_pred[i] == 'spam':  # Positive
                if np.array(self.y_true)[i] == 'spam':
                    tp += 1
                else:
                    fp += 1
            else:
                if np.array(self.y_true)[i] == 'spam':
                    fn += 1
                else:
                    tn += 1
        return tp, fp, fn, tn

    def raw_accuracy(self, y_pred):
        """
        Calculate the accuracy of the classifier
        """
        correct = 0
        total = 0
        y_test = np.array(self.y_true)
        for i in range(len(y_pred)):
            if y_test[i] == y_pred[i]:
                correct += 1
            total += 1

        return correct / total

    def precision(self):
        """
        Calculate the precision of the classifier
        """
        return self.tp / (self.tp + self.fp)

    def recall(self):
        """
        Calculate the recall of the classifier
        """
        return self.tp / (self.tp + self.fn)
