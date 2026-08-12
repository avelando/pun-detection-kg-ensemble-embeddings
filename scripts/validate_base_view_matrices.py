import numpy as np

from pun_detection.base_views import (
    BASE_VIEW_NAMES,
    generate_base_view_matrices,
)
from pun_detection.config import (
    EXPERIMENT,
)
from pun_detection.data import (
    load_development_splits,
)


def print_matrix_summary(
    name: str,
    matrix: np.ndarray,
) -> None:
    print(
        f"{name}: "
        f"shape={matrix.shape}, "
        f"dtype={matrix.dtype}, "
        f"min_probability="
        f"{matrix.min():.6f}, "
        f"max_probability="
        f"{matrix.max():.6f}"
    )

    for column_index, column_name in enumerate(
        BASE_VIEW_NAMES
    ):
        probabilities = matrix[
            :,
            column_index,
        ]

        print(
            f"{name}/{column_name}: "
            f"mean={probabilities.mean():.6f}, "
            f"std={probabilities.std():.6f}, "
            f"min={probabilities.min():.6f}, "
            f"max={probabilities.max():.6f}"
        )


def main():
    splits = load_development_splits()

    first_run = generate_base_view_matrices(
        train=splits.train,
        validation=splits.validation,
        seed=EXPERIMENT.primary_seed,
    )

    second_run = generate_base_view_matrices(
        train=splits.train,
        validation=splits.validation,
        seed=EXPERIMENT.primary_seed,
    )

    if first_run.columns != BASE_VIEW_NAMES:
        raise ValueError(
            "Base view column order is invalid"
        )

    if (
        first_run.selected_embedding_model
        != second_run.selected_embedding_model
    ):
        raise ValueError(
            "Selected embedding changed "
            "between runs"
        )

    if not np.array_equal(
        first_run.train_oof,
        second_run.train_oof,
    ):
        raise ValueError(
            "Train OOF base view matrix "
            "is not deterministic"
        )

    if not np.array_equal(
        first_run.validation,
        second_run.validation,
    ):
        raise ValueError(
            "Validation base view matrix "
            "is not deterministic"
        )

    print(
        f"selected_embedding_model="
        f"{first_run.selected_embedding_model}"
    )

    print(
        "columns="
        + ",".join(
            first_run.columns
        )
    )

    print_matrix_summary(
        "train_oof",
        first_run.train_oof,
    )

    print_matrix_summary(
        "validation",
        first_run.validation,
    )

    print(
        "Base view matrices are valid"
    )


if __name__ == "__main__":
    main()