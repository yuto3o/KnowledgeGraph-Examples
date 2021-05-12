# -*- coding: utf-8 -*-
import torch


def evaluate_fact_classification(y_pred: torch.LongTensor, y_true: torch.LongTensor, labels: torch.LongTensor = None):
    if labels is None:
        labels = torch.unique(torch.cat([y_pred, y_true]), sorted=True)

    num_classes = len(labels)
    confusion_matrix = torch.ones(num_classes, 2, 2)
    for i, label in enumerate(labels):
        y_pred_class_i = torch.eq(y_pred, label)
        y_true_class_i = torch.eq(y_true, label)

        confusion_matrix[i, 0, 0] = torch.sum(~y_pred_class_i & ~y_true_class_i)  # TN
        confusion_matrix[i, 1, 0] = torch.sum(y_pred_class_i & ~y_true_class_i)  # FP
        confusion_matrix[i, 0, 1] = torch.sum(~y_pred_class_i & y_true_class_i)  # FN
        confusion_matrix[i, 1, 1] = torch.sum(y_pred_class_i & y_true_class_i)  # TP

    p = torch.zeros(num_classes)
    r = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    tp = 0.
    fp = 0.
    fn = 0.

    for i in range(num_classes):
        p[i] = _divide(confusion_matrix[i, 1, 1], confusion_matrix[i, 1, 1] + confusion_matrix[i, 1, 0])
        r[i] = _divide(confusion_matrix[i, 1, 1], confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 1])
        f1[i] = _divide(2 * p[i] * r[i], p[i] + r[i])

        tp += confusion_matrix[i, 1, 1]
        fp += confusion_matrix[i, 1, 0]
        fn += confusion_matrix[i, 0, 1]

    p_micro = _divide(tp, tp + fp)
    r_micro = _divide(tp, tp + fn)
    f1_micro = _divide(2 * p_micro * r_micro, p_micro + r_micro)

    p_macro = torch.mean(p)
    r_macro = torch.mean(r)
    f1_macro = torch.mean(f1)

    return {'micro': {'p': p_micro, 'r': r_micro, 'f1': f1_micro},
            'macro': {'p': p_macro, 'r': r_macro, 'f1': f1_macro},
            'p': p, 'r': r, 'f1': f1}


def _divide(x, y):
    # handles divide-by-zero
    mask = y == 0.0
    y = y.clone()
    y[mask] = 1.
    return x / y
