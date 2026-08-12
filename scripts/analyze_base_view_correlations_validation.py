import json

import numpy as np
import pandas as pd

from pun_detection.base_view_cache import (
    load_base_view_cache,
)
from pun_detection.base_views import (
    BASE_VIEW_NAMES,
)
from pun_detection.config import (
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.fingerprints import (
    array_fingerprint,
)


def validate_correlation_matrix(
    matrix: np.ndarray,
) -> None:
    expected_shape = (
        len(BASE_VIEW_NAMES),
        len(BASE_VIEW_NAMES),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            "Correlation matrix has invalid shape"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            "Correlation matrix contains "
            "non-finite values"
        )

    if not np.allclose(
        np.diag(matrix),
        1.0,
    ):
        raise ValueError(
            "Correlation matrix has invalid diagonal"
        )

    if not np.allclose(
        matrix,
        matrix.T,
    ):
        raise ValueError(
            "Correlation matrix is not symmetric"
        )


def correlation_matrix(
    values: np.ndarray,
) -> np.ndarray:
    matrix = np.corrcoef(
        values,
        rowvar=False,
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    validate_correlation_matrix(
        matrix
    )

    return matrix


def main():
    splits = load_development_splits()

    train = splits.train
    validation = splits.validation

    train_correlations = []
    validation_correlations = []

    per_seed = {}

    selected_embedding_models = set()

    for seed in EXPERIMENT.seeds:
        matrices = load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )

        selected_embedding_models.add(
            matrices.selected_embedding_model
        )

        train_correlation = correlation_matrix(
            matrices.train_oof
        )

        validation_correlation = (
            correlation_matrix(
                matrices.validation
            )
        )

        train_correlations.append(
            train_correlation
        )

        validation_correlations.append(
            validation_correlation
        )

        per_seed[
            str(seed)
        ] = {
            "train_oof_fingerprint": (
                array_fingerprint(
                    matrices.train_oof
                )
            ),
            "validation_fingerprint": (
                array_fingerprint(
                    matrices.validation
                )
            ),
            "train_oof_correlation": (
                train_correlation.tolist()
            ),
            "validation_correlation": (
                validation_correlation.tolist()
            ),
        }

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Base view caches use different "
            "selected embedding models"
        )

    train_stack = np.stack(
        train_correlations,
        axis=0,
    )

    validation_stack = np.stack(
        validation_correlations,
        axis=0,
    )

    train_mean = train_stack.mean(
        axis=0
    )

    train_std = train_stack.std(
        axis=0,
        ddof=1,
    )

    validation_mean = (
        validation_stack.mean(
            axis=0
        )
    )

    validation_std = (
        validation_stack.std(
            axis=0,
            ddof=1,
        )
    )

    output = {
        "analysis_type": (
            "base_view_correlation_analysis"
        ),
        "selection_role": "diagnostic_only",
        "method": "pearson",
        "views": list(
            BASE_VIEW_NAMES
        ),
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "selected_embedding_model": next(
            iter(
                selected_embedding_models
            )
        ),
        "per_seed": per_seed,
        "summary": {
            "train_oof": {
                "mean": train_mean.tolist(),
                "std": train_std.tolist(),
            },
            "validation": {
                "mean": (
                    validation_mean.tolist()
                ),
                "std": (
                    validation_std.tolist()
                ),
            },
        },
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        PATHS.validation_results_dir
        / "base_view_correlations.json"
    )

    train_csv_path = (
        PATHS.validation_results_dir
        / "base_view_train_oof_correlations.csv"
    )

    validation_csv_path = (
        PATHS.validation_results_dir
        / "base_view_validation_correlations.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    pd.DataFrame(
        train_mean,
        index=BASE_VIEW_NAMES,
        columns=BASE_VIEW_NAMES,
    ).to_csv(
        train_csv_path
    )

    pd.DataFrame(
        validation_mean,
        index=BASE_VIEW_NAMES,
        columns=BASE_VIEW_NAMES,
    ).to_csv(
        validation_csv_path
    )

    print(
        "Train OOF mean Pearson correlations"
    )

    print(
        pd.DataFrame(
            train_mean,
            index=BASE_VIEW_NAMES,
            columns=BASE_VIEW_NAMES,
        ).round(6)
    )

    print()

    print(
        "Validation mean Pearson correlations"
    )

    print(
        pd.DataFrame(
            validation_mean,
            index=BASE_VIEW_NAMES,
            columns=BASE_VIEW_NAMES,
        ).round(6)
    )

    print()

    for left_index, left_name in enumerate(
        BASE_VIEW_NAMES
    ):
        for right_index in range(
            left_index + 1,
            len(BASE_VIEW_NAMES),
        ):
            right_name = BASE_VIEW_NAMES[
                right_index
            ]

            print(
                f"{left_name} vs {right_name}: "
                f"train="
                f"{train_mean[left_index, right_index]:.6f}"
                f"±"
                f"{train_std[left_index, right_index]:.6f}, "
                f"validation="
                f"{validation_mean[left_index, right_index]:.6f}"
                f"±"
                f"{validation_std[left_index, right_index]:.6f}"
            )

    print(
        f"Saved correlation analysis to "
        f"{json_path}"
    )


if __name__ == "__main__":
    main()