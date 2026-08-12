from dataclasses import dataclass

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pun_detection.config import (
    DATA,
    EXPERIMENT,
    REFERENCE_BASELINE,
)


@dataclass
class ReferenceTfidfModel:
    vectorizer: TfidfVectorizer
    classifier: VotingClassifier


def make_reference_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords.words("portuguese"),
    )


def make_reference_tfidf_classifier(
    seed: int,
) -> VotingClassifier:
    random_forest = RandomForestClassifier(
        n_estimators=REFERENCE_BASELINE.rf_estimators,
        criterion=REFERENCE_BASELINE.rf_criterion,
        max_depth=REFERENCE_BASELINE.rf_max_depth,
        random_state=seed,
        n_jobs=1,
    )

    logistic_regression = LogisticRegression(
        random_state=seed,
        max_iter=REFERENCE_BASELINE.lr_max_iter,
    )

    svm = SVC(
        C=REFERENCE_BASELINE.svm_c,
        kernel=REFERENCE_BASELINE.svm_kernel,
        probability=True,
        random_state=seed,
    )

    return VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("lr", logistic_regression),
            ("svm", svm),
        ],
        voting=REFERENCE_BASELINE.voting,
        n_jobs=1,
    )


def fit_reference_tfidf_model(
    train: pd.DataFrame,
    seed: int = EXPERIMENT.primary_seed,
) -> ReferenceTfidfModel:
    vectorizer = make_reference_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(
        train[
            DATA.text_column
        ].astype(str)
    )

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    classifier = make_reference_tfidf_classifier(
        seed=seed,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    return ReferenceTfidfModel(
        vectorizer=vectorizer,
        classifier=classifier,
    )


def predict_reference_tfidf_probabilities(
    model: ReferenceTfidfModel,
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