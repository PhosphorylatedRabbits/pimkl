import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from collections import defaultdict
from scipy.stats import rankdata


def roc_two_classes(y_test, y_score):
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    fpr["class"], tpr["class"], threshold["class"] = roc_curve(
        y_test[:, 1], y_score[:, 1]
    )
    roc_auc["class"] = auc(fpr["class"], tpr["class"])
    return fpr, tpr, threshold, roc_auc


def roc_multiclass(y_test, y_score, n_classes=None):
    if n_classes is None:
        n_classes = y_score.shape[1]
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], threshold['micro'] = roc_curve(
        y_test.ravel(), y_score.ravel()
    )
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    mean_threshold = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_threshold += interp(all_fpr, fpr[i], threshold[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    mean_threshold /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    threshold["macro"] = mean_threshold
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, threshold, roc_auc


def roc_analysis(y_test, y_score):
    n_classes = y_score.shape[1]
    if n_classes > 2:
        return roc_multiclass(y_test, y_score, n_classes)
    else:
        return roc_two_classes(y_test, y_score)


def performances(y_true, y_score):
    n_classes = y_score.shape[1]
    y_predicted = np.array(
        np.apply_along_axis(rankdata, 1, y_score) >= n_classes, dtype=np.int
    )
    tp = defaultdict(int)
    tn = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for a_class in range(n_classes):
        for sample in range(y_true.shape[0]):
            if (
                y_predicted[sample, a_class] == 1
                and y_true[sample, a_class] == 1
            ):
                tp[a_class] += 1.
            elif (
                y_predicted[sample, a_class] == 1
                and y_true[sample, a_class] == 0
            ):
                fp[a_class] += 1.
            elif (
                y_predicted[sample, a_class] == 0
                and y_true[sample, a_class] == 0
            ):
                tn[a_class] += 1.
            elif (
                y_predicted[sample, a_class] == 0
                and y_true[sample, a_class] == 1
            ):
                fn[a_class] += 1.
    return tp, tn, fp, fn


def sensitivity(tp, fn):
    return tp / float(tp + fn)


def specificity(tn, fp):
    return tn / float(tn + fp)
