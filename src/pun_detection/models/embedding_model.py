from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

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
    make_logistic_classifier,
)


@dataclass
class EmbeddingViewModel:
    model_name: str
    classifier: LogisticRegression
    iterations: int


def fit_embedding_view_model(
    train: pd.DataFrame,
    model_name: str,
    seed: int = EXPERIMENT.primary_seed,
) -> EmbeddingViewModel:
    get_embedding_config(
        model_name
    )

    X_train = load_embedding_cache(
        model_name=model_name,
        split_name="train",
        dataframe=train,
    )

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    classifier = make_logistic_classifier(
        seed=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            ConvergenceWarning,
        )

        classifier.fit(
            X_train,
            y_train,
        )

    iterations = int(
        np.max(
            classifier.n_iter_
        )
    )

    return EmbeddingViewModel(
        model_name=model_name,
        classifier=classifier,
        iterations=iterations,
    )


def predict_embedding_view_probabilities(
    model: EmbeddingViewModel,
    dataframe: pd.DataFrame,
    split_name: str,
) -> np.ndarray:
    X = load_embedding_cache(
        model_name=model.model_name,
        split_name=split_name,
        dataframe=dataframe,
    )

    probabilities = (
        model.classifier.predict_proba(
            X
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
            f"{model.model_name} produced "
            f"non-finite probabilities"
        )

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(
            f"{model.model_name} produced "
            f"invalid probabilities"
        )

    return probabilities