from dataclasses import dataclass

import numpy as np
import pandas as pd

from pun_detection.config import (
    DATA,
    EXPERIMENT,
)
from pun_detection.embedding_cache import (
    load_embedding_cache,
)
from pun_detection.embeddings import (
    get_embedding_config,
)
from pun_detection.models.base import (
    fit_logistic_classifier,
)
from pun_detection.oof import (
    create_oof_splits,
)


@dataclass(frozen=True)
class EmbeddingOOFPredictions:
    model_name: str
    probabilities: np.ndarray
    fold_iterations: tuple[int, ...]


def generate_embedding_oof_predictions(
    train: pd.DataFrame,
    model_name: str,
    seed: int = EXPERIMENT.primary_seed,
) -> EmbeddingOOFPredictions:
    get_embedding_config(
        model_name
    )

    embeddings = load_embedding_cache(
        model_name=model_name,
        split_name="train",
        dataframe=train,
    )

    y = train[
        DATA.label_column
    ].astype(int).to_numpy()

    oof_predictions = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    fold_iterations = []

    oof_splits = create_oof_splits(
        dataframe=train,
    )

    for split in oof_splits:
        X_train = embeddings[
            split.train_indices
        ]

        X_holdout = embeddings[
            split.holdout_indices
        ]

        y_train = y[
            split.train_indices
        ]

        classifier = fit_logistic_classifier(
            X=X_train,
            y=y_train,
            seed=seed,
        )

        probabilities = (
            classifier.predict_proba(
                X_holdout
            )[:, 1]
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                f"{model_name} produced non-finite "
                f"OOF probabilities in fold {split.fold}"
            )

        if (
            np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ValueError(
                f"{model_name} produced invalid "
                f"OOF probabilities in fold {split.fold}"
            )

        oof_predictions[
            split.holdout_indices
        ] = probabilities

        fold_iterations.append(
            int(
                np.max(
                    classifier.n_iter_
                )
            )
        )

    if np.isnan(
        oof_predictions
    ).any():
        raise ValueError(
            f"{model_name} contains missing "
            "OOF predictions"
        )

    if not np.isfinite(
        oof_predictions
    ).all():
        raise ValueError(
            f"{model_name} contains non-finite "
            "OOF predictions"
        )

    if (
        np.any(oof_predictions < 0.0)
        or np.any(oof_predictions > 1.0)
    ):
        raise ValueError(
            f"{model_name} contains invalid "
            "OOF probabilities"
        )

    if len(
        fold_iterations
    ) != EXPERIMENT.oof_folds:
        raise ValueError(
            f"{model_name} has an invalid number "
            "of OOF fold iteration counts"
        )

    return EmbeddingOOFPredictions(
        model_name=model_name,
        probabilities=oof_predictions,
        fold_iterations=tuple(
            fold_iterations
        ),
    )