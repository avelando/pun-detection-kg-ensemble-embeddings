from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pun_detection.config import DATA, EXPERIMENT
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.features import (
    fit_graph_encoder_set,
    transform_graph_encoder_set,
)
from pun_detection.models.base import make_logistic_classifier
from pun_detection.oof import create_oof_splits


GRAPH_NAMES = (
    "cooccurrence",
    "ppmi",
    "pun_context",
)


@dataclass(frozen=True)
class GraphOOFPredictions:
    cooccurrence: np.ndarray
    ppmi: np.ndarray
    pun_context: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "cooccurrence": self.cooccurrence,
            "ppmi": self.ppmi,
            "pun_context": self.pun_context,
        }


def generate_graph_oof_predictions(
    train: pd.DataFrame,
    seed: int = EXPERIMENT.primary_seed,
) -> GraphOOFPredictions:
    oof_predictions = {
        name: np.full(
            len(train),
            np.nan,
            dtype=np.float64,
        )
        for name in GRAPH_NAMES
    }

    oof_splits = create_oof_splits(
        dataframe=train,
        seed=seed,
    )

    for split in oof_splits:
        fold_train = train.iloc[
            split.train_indices
        ]

        fold_holdout = train.iloc[
            split.holdout_indices
        ]

        graph_set = build_graph_set(
            fold_train
        )

        encoders = fit_graph_encoder_set(
            graph_set,
            seed=seed,
        )

        train_features = (
            transform_graph_encoder_set(
                fold_train[
                    DATA.text_column
                ].tolist(),
                encoders,
            )
        )

        holdout_features = (
            transform_graph_encoder_set(
                fold_holdout[
                    DATA.text_column
                ].tolist(),
                encoders,
            )
        )

        y_train = fold_train[
            DATA.label_column
        ].astype(int).to_numpy()

        for graph_name in GRAPH_NAMES:
            X_train = train_features.as_dict()[
                graph_name
            ]

            X_holdout = holdout_features.as_dict()[
                graph_name
            ]

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_holdout_scaled = scaler.transform(
                X_holdout
            )

            classifier = make_logistic_classifier(
                seed=seed,
            )

            classifier.fit(
                X_train_scaled,
                y_train,
            )

            probabilities = classifier.predict_proba(
                X_holdout_scaled
            )[:, 1]

            oof_predictions[
                graph_name
            ][split.holdout_indices] = (
                probabilities
            )

    for graph_name, probabilities in (
        oof_predictions.items()
    ):
        if np.isnan(probabilities).any():
            raise ValueError(
                f"{graph_name} contains missing "
                f"OOF predictions"
            )

    return GraphOOFPredictions(
        cooccurrence=oof_predictions[
            "cooccurrence"
        ],
        ppmi=oof_predictions[
            "ppmi"
        ],
        pun_context=oof_predictions[
            "pun_context"
        ],
    )