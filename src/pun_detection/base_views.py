from dataclasses import dataclass

import numpy as np
import pandas as pd

from pun_detection.models.embedding_model import (
    fit_embedding_view_model,
    predict_embedding_view_probabilities,
)
from pun_detection.models.embedding_oof import (
    generate_embedding_oof_predictions,
)
from pun_detection.models.graph_model import (
    fit_graph_view_model,
    predict_graph_view_probabilities,
)
from pun_detection.models.graph_oof import (
    generate_graph_oof_predictions,
)
from pun_detection.models.tfidf_model import (
    fit_tfidf_view_model,
    predict_tfidf_view_probabilities,
)
from pun_detection.models.tfidf_oof import (
    generate_tfidf_oof_predictions,
)
from pun_detection.selection import (
    get_selected_embedding_model,
)


BASE_VIEW_NAMES = (
    "tfidf",
    "selected_embedding",
    "cooccurrence",
    "ppmi",
    "pun_context",
)


BASE_VIEW_INDEX = {
    view_name: index
    for index, view_name in enumerate(
        BASE_VIEW_NAMES
    )
}


def validate_base_view_names(
    view_names,
) -> tuple[str, ...]:
    normalized = tuple(
        view_names
    )

    if not normalized:
        raise ValueError(
            "Base view selection cannot be empty"
        )

    if len(
        normalized
    ) != len(
        set(normalized)
    ):
        raise ValueError(
            "Base view selection contains duplicates"
        )

    unknown = [
        view_name
        for view_name in normalized
        if view_name not in BASE_VIEW_INDEX
    ]

    if unknown:
        raise ValueError(
            f"Unknown base views: {unknown}"
        )

    return normalized


def select_base_view_matrix(
    matrix: np.ndarray,
    view_names,
) -> np.ndarray:
    view_names = validate_base_view_names(
        view_names
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Base view matrix must be two-dimensional"
        )

    if matrix.shape[1] != len(
        BASE_VIEW_NAMES
    ):
        raise ValueError(
            "Base view matrix has an invalid "
            "number of columns"
        )

    indices = [
        BASE_VIEW_INDEX[
            view_name
        ]
        for view_name in view_names
    ]

    selected = matrix[
        :,
        indices,
    ]

    if not np.isfinite(
        selected
    ).all():
        raise ValueError(
            "Selected base view matrix contains "
            "non-finite values"
        )

    if (
        np.any(selected < 0.0)
        or np.any(selected > 1.0)
    ):
        raise ValueError(
            "Selected base view matrix contains "
            "invalid probabilities"
        )

    return selected


@dataclass(frozen=True)
class BaseViewMatrices:
    selected_embedding_model: str
    train_oof: np.ndarray
    validation: np.ndarray
    columns: tuple[str, ...]


def validate_base_view_matrix(
    matrix: np.ndarray,
    expected_rows: int,
    matrix_name: str,
) -> None:
    if matrix.shape != (
        expected_rows,
        len(BASE_VIEW_NAMES),
    ):
        raise ValueError(
            f"{matrix_name} has invalid shape "
            f"{matrix.shape}"
        )

    if matrix.dtype != np.float64:
        raise ValueError(
            f"{matrix_name} has invalid dtype "
            f"{matrix.dtype}"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            f"{matrix_name} contains "
            "non-finite probabilities"
        )

    if (
        np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ValueError(
            f"{matrix_name} contains "
            "invalid probabilities"
        )


def generate_base_view_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> BaseViewMatrices:
    selected_embedding_model = (
        get_selected_embedding_model(
            train=train,
            validation=validation,
        )
    )

    tfidf_oof = (
        generate_tfidf_oof_predictions(
            train=train,
            seed=seed,
        )
    )

    embedding_oof = (
        generate_embedding_oof_predictions(
            train=train,
            model_name=selected_embedding_model,
            seed=seed,
        )
    )

    graph_oof = (
        generate_graph_oof_predictions(
            train=train,
            seed=seed,
        )
    )

    train_oof = np.column_stack(
        (
            tfidf_oof,
            embedding_oof.probabilities,
            graph_oof.cooccurrence,
            graph_oof.ppmi,
            graph_oof.pun_context,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    tfidf_model = fit_tfidf_view_model(
        train=train,
        seed=seed,
    )

    tfidf_validation = (
        predict_tfidf_view_probabilities(
            model=tfidf_model,
            dataframe=validation,
        )
    )

    embedding_model = (
        fit_embedding_view_model(
            train=train,
            model_name=selected_embedding_model,
            seed=seed,
        )
    )

    embedding_validation = (
        predict_embedding_view_probabilities(
            model=embedding_model,
            dataframe=validation,
            split_name="validation",
        )
    )

    graph_model = fit_graph_view_model(
        train=train,
        seed=seed,
    )

    graph_validation = (
        predict_graph_view_probabilities(
            model=graph_model,
            dataframe=validation,
        )
    )

    validation_matrix = np.column_stack(
        (
            tfidf_validation,
            embedding_validation,
            graph_validation[
                "cooccurrence"
            ],
            graph_validation[
                "ppmi"
            ],
            graph_validation[
                "pun_context"
            ],
        )
    ).astype(
        np.float64,
        copy=False,
    )

    validate_base_view_matrix(
        matrix=train_oof,
        expected_rows=len(train),
        matrix_name="train_oof",
    )

    validate_base_view_matrix(
        matrix=validation_matrix,
        expected_rows=len(validation),
        matrix_name="validation",
    )

    return BaseViewMatrices(
        selected_embedding_model=(
            selected_embedding_model
        ),
        train_oof=train_oof,
        validation=validation_matrix,
        columns=BASE_VIEW_NAMES,
    )