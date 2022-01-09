"""
@author: Jaime de Boer
"""
import pandas as pd
from NaiveBayes import NaiveBayes
from sklearn.model_selection import KFold
from PerformanceMetrics import PerformanceMetrics
from AttributeWeightingIG import AttributeWeightingIG
from AttributeWeightingGR import AttributeWeightingGR

if __name__ == '__main__':

    data = pd.read_csv('spam.csv')

    # Put data in X and y labels, and separate into test and train data
    X = data['v2']
    y = data['v1']

    k = 4
    kf = KFold(n_splits=k, random_state=None)

    accuracy = []
    precision = []
    recall = []

    tp = []
    fp = []
    tn = []
    fn = []

    # K-Fold Cross Validation with all three algorithms
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for i in range(3):
            if i == 0:
                alg = NaiveBayes(X_train, y_train)
                y_pred = alg.parse_messages(X_test)
            elif i == 1:
                alg = AttributeWeightingIG(X_train, y_train)
                y_pred = alg.parse_messages(X_test)
            else:
                alg = AttributeWeightingGR(X_train, y_train)
                y_pred = alg.parse_messages(X_test)

            # Save & calulate metrics
            pm = PerformanceMetrics(y_test, y_pred)
            accuracy.append(pm.raw_accuracy(y_pred))
            precision.append(pm.precision())
            recall.append(pm.recall())

            tp.append(pm.tp)
            fp.append(pm.fp)
            tn.append(pm.tn)
            fn.append(pm.fn)

    # Get metrics ready for printing
    accuracy_nb = 0
    accuracy_ig = 0
    accuracy_gr = 0

    precision_nb = 0
    precision_ig = 0
    precision_gr = 0

    recall_nb = 0
    recall_ig = 0
    recall_gr = 0

    tp_nb = 0
    fp_nb = 0
    tn_nb = 0
    fn_nb = 0

    tp_ig = 0
    fp_ig = 0
    tn_ig = 0
    fn_ig = 0

    tp_gr = 0
    fp_gr = 0
    tn_gr = 0
    fn_gr = 0

    for i in range(k*3):
        if i % 3 == 0:
            accuracy_nb += accuracy[i] * 1/k
            precision_nb += precision[i] * 1/k
            recall_nb += recall[i] * 1/k

            tp_nb += tp[i] * 1/k
            fp_nb += fp[i] * 1/k
            tn_nb += tn[i] * 1/k
            fn_nb += fn[i] * 1/k

        elif i % 3 == 1:
            accuracy_ig += accuracy[i] * 1/k
            precision_ig += precision[i] * 1/k
            recall_ig += recall[i] * 1/k

            tp_ig += tp[i] * 1/k
            fp_ig += fp[i] * 1/k
            tn_ig += tn[i] * 1/k
            fn_ig += fn[i] * 1/k
        else:
            accuracy_gr += accuracy[i] * 1/k
            precision_gr += precision[i] * 1/k
            recall_gr += recall[i] * 1/k

            tp_gr += tp[i] * 1/k
            fp_gr += fp[i] * 1/k
            tn_gr += tn[i] * 1/k
            fn_gr += fn[i] * 1/k

    # Print results
    print("Naive Bayes:")
    print("Accuracy", accuracy_nb)
    print("Precision", precision_nb)
    print("Recall", recall_nb)

    print("\nConfusion matrix values")
    print("True Positives", tp_nb)
    print("False Positives", fp_nb)
    print("True Negatives", tn_nb)
    print("False Negatives", fn_nb)

    print("\n\nInformation Gain:")
    print("Accuracy", accuracy_ig)
    print("Precision", precision_ig)
    print("Recall", recall_ig)

    print("\nConfusion matrix values")
    print("True Positives", tp_ig)
    print("False Positives", fp_ig)
    print("True Negatives", tn_ig)
    print("False Negatives", fn_ig)

    print("\n\nGain Ratio:")
    print("Accuracy", accuracy_gr)
    print("Precision", precision_gr)
    print("Recall", recall_gr)

    print("\nConfusion matrix values")
    print("True Positives", tp_gr)
    print("False Positives", fp_gr)
    print("True Negatives", tn_gr)
    print("False Negatives", fn_gr)
