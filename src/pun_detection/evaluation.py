from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def probabilities_to_predictions(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain non-finite values"
        )

    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError(
            "Probabilities must be between zero and one"
        )

    return (
        probabilities >= threshold
    ).astype(int)


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    predictions = probabilities_to_predictions(
        probabilities,
        threshold=threshold,
    )

    if len(y_true) != len(predictions):
        raise ValueError(
            "Targets and predictions have different sizes"
        )

    return BinaryMetrics(
        accuracy=float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        macro_precision=float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        macro_recall=float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        samples=len(y_true),
    )


def summarize_metric_values(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "Metric values cannot be empty"
        )

    standard_deviation = (
        float(
            np.std(
                array,
                ddof=1,
            )
        )
        if array.size > 1
        else 0.0
    )

    return {
        "mean": float(array.mean()),
        "std": standard_deviation,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_binary_metrics(
    metrics: list[BinaryMetrics],
) -> dict[str, dict[str, float]]:
    if not metrics:
        raise ValueError(
            "Metrics cannot be empty"
        )

    return {
        "accuracy": summarize_metric_values(
            [
                metric.accuracy
                for metric in metrics
            ]
        ),
        "macro_precision": summarize_metric_values(
            [
                metric.macro_precision
                for metric in metrics
            ]
        ),
        "macro_recall": summarize_metric_values(
            [
                metric.macro_recall
                for metric in metrics
            ]
        ),
        "macro_f1": summarize_metric_values(
            [
                metric.macro_f1
                for metric in metrics
            ]
        ),
    }