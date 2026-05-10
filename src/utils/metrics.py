"""Evaluation metrics for binary image detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Clipping avoids overflow in exp() during sigmoid conversion from large-magnitude logits.
_LOGIT_CLIP_MIN = -500.0
_LOGIT_CLIP_MAX = 500.0


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return accuracy, ROC-AUC, and confusion matrix for binary predictions."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_true_arr.size == 0:
        raise ValueError("y_true must contain at least one ground-truth label.")
    if y_pred_arr.size == 0:
        raise ValueError("y_pred must contain at least one prediction.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of elements.")

    # Treat predictions outside [0, 1] as logits and map them to probabilities.
    if np.any((y_pred_arr < 0) | (y_pred_arr > 1)):
        clipped_logits = np.clip(y_pred_arr, _LOGIT_CLIP_MIN, _LOGIT_CLIP_MAX)
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
