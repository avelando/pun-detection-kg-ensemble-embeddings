import numpy as np

from pun_detection.base_views import (
    BASE_VIEW_NAMES,
)


def soft_voting_probabilities(
    view_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        view_matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Soft voting input must be two-dimensional"
        )

    if matrix.shape[1] != len(
        BASE_VIEW_NAMES
    ):
        raise ValueError(
            "Soft voting input has an invalid "
            "number of base views"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            "Soft voting input contains "
            "non-finite probabilities"
        )

    if (
        np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ValueError(
            "Soft voting input contains "
            "invalid probabilities"
        )

    probabilities = np.mean(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Soft voting produced "
            "non-finite probabilities"
        )

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(
            "Soft voting produced "
            "invalid probabilities"
        )

    return probabilities