# src/utils/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error
)


def mean_absolute_scaled_error(y_true, y_pred, y_train_mean=None):
    """
    Computes the Mean Absolute Scaled Error (MASE).
    If `y_train_mean` is not provided, it uses the in-sample naive forecast.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = y_true.shape[0]

    if y_train_mean is None:
        d = np.abs(np.diff(y_true)).mean()
    else:
        d = y_train_mean

    errors = np.abs(y_true - y_pred)
    return errors.mean() / d if d != 0 else np.nan


def classification_report_dict(y_true, y_pred, y_prob=None):
    """
    Generate a dictionary of classification metrics.
    Optionally includes ROC AUC if probabilities are provided.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for the positive class.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='binary', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics_dict["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics_dict["roc_auc"] = None

    metrics_dict["classification_report"] = report
    return metrics_dict
