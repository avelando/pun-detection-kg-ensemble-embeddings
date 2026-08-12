from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pun_detection.config import (
    EXPERIMENT,
)
from pun_detection.models.reference_tfidf import (
    make_reference_tfidf_classifier,
    make_reference_tfidf_vectorizer,
)


def main():
    vectorizer = make_reference_tfidf_vectorizer()

    if vectorizer.ngram_range != (1, 2):
        raise ValueError(
            "Invalid reference TF-IDF ngram range"
        )

    if vectorizer.strip_accents is not None:
        raise ValueError(
            "Reference TF-IDF must not strip accents"
        )

    classifier = make_reference_tfidf_classifier(
        EXPERIMENT.primary_seed
    )

    estimators = dict(
        classifier.estimators
    )

    random_forest = estimators["rf"]
    logistic_regression = estimators["lr"]
    svm = estimators["svm"]

    if not isinstance(
        random_forest,
        RandomForestClassifier,
    ):
        raise ValueError(
            "Invalid reference Random Forest"
        )

    if random_forest.n_estimators != 100:
        raise ValueError(
            "Invalid Random Forest estimator count"
        )

    if random_forest.criterion != "entropy":
        raise ValueError(
            "Invalid Random Forest criterion"
        )

    if random_forest.max_depth != 15:
        raise ValueError(
            "Invalid Random Forest maximum depth"
        )

    if not isinstance(
        logistic_regression,
        LogisticRegression,
    ):
        raise ValueError(
            "Invalid reference Logistic Regression"
        )

    if logistic_regression.max_iter != 2000:
        raise ValueError(
            "Invalid Logistic Regression iterations"
        )

    if not isinstance(
        svm,
        SVC,
    ):
        raise ValueError(
            "Invalid reference SVM"
        )

    if svm.kernel != "rbf":
        raise ValueError(
            "Invalid SVM kernel"
        )

    if not svm.probability:
        raise ValueError(
            "Reference SVM must produce probabilities"
        )

    if classifier.voting != "soft":
        raise ValueError(
            "Reference ensemble must use soft voting"
        )

    print(
        "Reference TF-IDF ensemble configuration is valid"
    )


if __name__ == "__main__":
    main()