"""Метрики для отчётов и Analytics-агента."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(y_true, y_pred, average: str = "macro"):
    """Возвращает (precision, recall, f1) — macro по умолчанию."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return float(p), float(r), float(f1)


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred, labels=[0, 1])