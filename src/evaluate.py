import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score
)


def metrics(y_true, y_pred, class_names, y_prob=None, inference_ms_per_image=None, checkpoint_path=None):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class": {},
    }
    total = cm.sum()
    for index, name in enumerate(class_names):
        tp = cm[index, index]
        fn = cm[index, :].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = total - tp - fn - fp
        result["per_class"][name] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        }
    if y_prob is not None:
        probabilities = np.asarray(y_prob, dtype=float)
        one_hot = np.eye(len(class_names))[np.asarray(y_true, dtype=int)]
        try:
            result["roc_auc_ovr_macro"] = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            result["roc_auc_ovr_macro"] = None
        try:
            result["pr_auc_macro"] = float(average_precision_score(one_hot, probabilities, average="macro"))
        except ValueError:
            result["pr_auc_macro"] = None
    if inference_ms_per_image is not None:
        result["inference_ms_per_image"] = float(inference_ms_per_image)
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            result["checkpoint_size_mb"] = round(path.stat().st_size / (1024 ** 2), 3)
    return result
