"""Evaluation metrics for binary image detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return accuracy, ROC-AUC, and confusion matrix for binary predictions."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_pred_arr.size == 0:
        raise ValueError("y_pred must contain at least one prediction.")

    # Treat predictions outside [0, 1] as logits and map them to probabilities.
    if np.any((y_pred_arr < 0) | (y_pred_arr > 1)):
        clipped_logits = np.clip(y_pred_arr, -500, 500)
        y_prob = 1.0 / (1.0 + np.exp(-clipped_logits))
    else:
        y_prob = y_pred_arr

    y_label = (y_prob >= 0.5).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_label)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true_arr, y_label),
    }
