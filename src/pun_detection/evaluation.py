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