from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pun_detection.config import DATA, EXPERIMENT
from pun_detection.models.base import fit_logistic_classifier
from pun_detection.text.features import make_tfidf_vectorizer


@dataclass
class TfidfViewModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


def fit_tfidf_view_model(
    train: pd.DataFrame,
    seed: int = EXPERIMENT.primary_seed,
) -> TfidfViewModel:
    vectorizer = make_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(
        train[
            DATA.text_column
        ].astype(str)
    )

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    classifier = fit_logistic_classifier(
        X=X_train,
        y=y_train,
        seed=seed,
    )

    return TfidfViewModel(
        vectorizer=vectorizer,
        classifier=classifier,
    )


def predict_tfidf_view_probabilities(
    model: TfidfViewModel,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    X = model.vectorizer.transform(
        dataframe[
            DATA.text_column
        ].astype(str)
    )

    return model.classifier.predict_proba(
        X
    )[:, 1]