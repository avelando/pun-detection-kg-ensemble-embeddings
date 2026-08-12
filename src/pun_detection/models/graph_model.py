from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pun_detection.config import DATA, EXPERIMENT
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.features import (
    GraphEncoderSet,
    fit_graph_encoder_set,
    transform_graph_encoder_set,
)
from pun_detection.models.base import fit_logistic_classifier


GRAPH_NAMES = (
    "cooccurrence",
    "ppmi",
    "pun_context",
)


@dataclass
class GraphViewModel:
    encoders: GraphEncoderSet
    scalers: dict[str, StandardScaler]
    classifiers: dict[str, LogisticRegression]


def fit_graph_view_model(
    train: pd.DataFrame,
    seed: int = EXPERIMENT.primary_seed,
) -> GraphViewModel:
    graph_set = build_graph_set(
        train
    )

    encoders = fit_graph_encoder_set(
        graph_set,
        seed=seed,
    )

    feature_set = transform_graph_encoder_set(
        train[
            DATA.text_column
        ].tolist(),
        encoders,
    )

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    scalers = {}
    classifiers = {}

    for graph_name in GRAPH_NAMES:
        X_train = feature_set.as_dict()[
            graph_name
        ]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        classifier = fit_logistic_classifier(
            X=X_train_scaled,
            y=y_train,
            seed=seed,
        )

        scalers[graph_name] = scaler
        classifiers[graph_name] = classifier

    return GraphViewModel(
        encoders=encoders,
        scalers=scalers,
        classifiers=classifiers,
    )


def predict_graph_view_probabilities(
    model: GraphViewModel,
    dataframe: pd.DataFrame,
) -> dict[str, np.ndarray]:
    feature_set = transform_graph_encoder_set(
        dataframe[
            DATA.text_column
        ].tolist(),
        model.encoders,
    )

    probabilities = {}

    for graph_name in GRAPH_NAMES:
        X = feature_set.as_dict()[
            graph_name
        ]

        X_scaled = model.scalers[
            graph_name
        ].transform(X)

        probabilities[graph_name] = (
            model.classifiers[
                graph_name
            ]
            .predict_proba(
                X_scaled
            )[:, 1]
        )

    return probabilities