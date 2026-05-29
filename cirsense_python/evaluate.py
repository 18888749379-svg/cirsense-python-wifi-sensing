import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for classification metrics.") from exc

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_rows_csv(rows: Iterable[Mapping], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to save.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
