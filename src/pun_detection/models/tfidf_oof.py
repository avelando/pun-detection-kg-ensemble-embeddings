import numpy as np
import pandas as pd

from pun_detection.config import DATA, EXPERIMENT
from pun_detection.models.base import make_logistic_classifier
from pun_detection.oof import create_oof_splits
from pun_detection.text.features import make_tfidf_vectorizer


def generate_tfidf_oof_predictions(
    train: pd.DataFrame,
    seed: int = EXPERIMENT.primary_seed,
) -> np.ndarray:
    oof_predictions = np.full(
        len(train),
        np.nan,
        dtype=np.float64,
    )

    oof_splits = create_oof_splits(
        dataframe=train,
    )

    for split in oof_splits:
        fold_train = train.iloc[
            split.train_indices
        ]

        fold_holdout = train.iloc[
            split.holdout_indices
        ]

        vectorizer = make_tfidf_vectorizer()

        X_train = vectorizer.fit_transform(
            fold_train[
                DATA.text_column
            ].astype(str)
        )

        X_holdout = vectorizer.transform(
            fold_holdout[
                DATA.text_column
            ].astype(str)
        )

        y_train = fold_train[
            DATA.label_column
        ].astype(int).to_numpy()

        classifier = make_logistic_classifier(
            seed=seed,
        )

        classifier.fit(
            X_train,
            y_train,
        )

        probabilities = classifier.predict_proba(
            X_holdout
        )[:, 1]

        oof_predictions[
            split.holdout_indices
        ] = probabilities

    if np.isnan(oof_predictions).any():
        raise ValueError(
            "TF-IDF contains missing OOF predictions"
        )

    return oof_predictions